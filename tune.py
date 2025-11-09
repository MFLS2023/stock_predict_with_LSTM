#!/usr/bin/env python
"""Optuna hyperparameter search runner for the PPO trading agent.

Workflow overview:
1. Sample learning_rate, n_steps, gamma, ent_coef (and derived batch_size) for each Optuna trial.
2. Train the PPO model via :func:`train_agent.train_ppo_model` and persist trial artifacts.
3. Execute a backtest with :func:`backtest.run_ppo_backtest` and use the Sharpe ratio as the objective score.
4. Optimise the study and export the best parameters, trial log, and model checkpoints.

Example:
    python tune.py --data data/sh510300.csv --valid-ratio 0.2 --trials 50 --total-timesteps 150000
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.trial import Trial

from backtest import run_ppo_backtest
from train_agent import train_ppo_model


@dataclass
class TuneArgs:
    data_path: Path
    valid_ratio: float
    total_timesteps: int
    trials: int
    study_name: str
    storage: Optional[str]
    output_dir: Path
    device: str
    seed: int
    initial_cash: float
    monthly_cash: float
    fee: float
    verbose: bool

MODEL_FILENAME = "ppo_model.zip"
SCALER_FILENAME = "feature_scaler.pkl"
TRIAL_SUBDIR = "trials"


def make_trial_logger(trial_number: int, verbose: bool):
    prefix = f"[Trial {trial_number:04d}]"

    def _logger(message: str) -> None:
        if verbose:
            print(f"{prefix} {message}")

    return _logger


def sample_trial_params(trial: Trial) -> Dict[str, Any]:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    gamma = trial.suggest_float("gamma", 0.94, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 1e-6, 0.02, log=True)
    n_steps = trial.suggest_categorical("n_steps", [256, 384, 512, 768, 1024, 1536])
    rollout_ratio = trial.suggest_categorical("rollout_factor", [2, 4, 8])
    batch_size = max(64, int(n_steps / rollout_ratio))
    batch_size = min(batch_size, int(n_steps))
    return {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "ent_coef": ent_coef,
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
    }


def parse_args(argv: Optional[list[str]] = None) -> TuneArgs:
    parser = argparse.ArgumentParser(description="Optuna tuning for PPO LSTM trading agent")
    parser.add_argument("--data", dest="data_path", required=True, type=Path, help="Path to CSV market data")
    parser.add_argument("--valid-ratio", type=float, default=0.2, help="Validation split ratio (0-0.5)")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Training timesteps per trial")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--study-name", type=str, default="ppo_lstm_tuning", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL, e.g. sqlite:///study.db")
    parser.add_argument("--output-dir", type=Path, default=Path("tuning_outputs"), help="Directory for study artifacts")
    parser.add_argument("--device", type=str, default="auto", help="Torch device for PPO (auto/cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed base")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Initial cash for environments")
    parser.add_argument("--monthly-cash", type=float, default=0.0, help="Monthly contribution for environments")
    parser.add_argument("--fee", type=float, default=0.001, help="Transaction fee ratio")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Optuna logging")

    args = parser.parse_args(argv)

    if not 0.0 < args.valid_ratio < 0.5:
        raise ValueError("--valid-ratio must be between 0 and 0.5 for a meaningful split")

    return TuneArgs(
        data_path=args.data_path,
        valid_ratio=args.valid_ratio,
        total_timesteps=args.total_timesteps,
        trials=args.trials,
        study_name=args.study_name,
        storage=args.storage,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        initial_cash=args.initial_cash,
        monthly_cash=args.monthly_cash,
        fee=args.fee,
        verbose=args.verbose,
    )


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def load_market_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[df["Date"].notna()].copy()
        df.sort_values("Date", inplace=True)
        df.set_index("Date", inplace=True)
    else:
        df.sort_index(inplace=True)
    if len(df) < 500:
        raise ValueError("Dataset is too small for tuning; need at least 500 rows")
    return df


def split_train_valid(df: pd.DataFrame, valid_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - valid_ratio))
    if split_idx <= 0 or split_idx >= len(df) - 50:
        raise ValueError("Validation split produced insufficient data; adjust --valid-ratio")
    train_df = df.iloc[:split_idx].copy()
    valid_df = df.iloc[split_idx:].copy()
    return train_df, valid_df


def run_study(args: TuneArgs, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> optuna.study.Study:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_root = args.output_dir / TRIAL_SUBDIR
    trial_root.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    optuna.logging.set_verbosity(optuna.logging.INFO if args.verbose else optuna.logging.WARNING)

    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=args.storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

    def objective(trial: Trial) -> float:
        seed = args.seed + trial.number
        set_global_seeds(seed)
        sampled_params = sample_trial_params(trial)
        ppo_kwargs: Dict[str, Any] = dict(sampled_params)
        ppo_kwargs["seed"] = seed

        trial_dir = trial_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        log_fn = make_trial_logger(trial.number, args.verbose)

        try:
            model_path = train_ppo_model(
                df_train=train_df.copy(),
                total_timesteps=args.total_timesteps,
                output_dir=trial_dir,
                model_filename=MODEL_FILENAME,
                device=args.device,
                log_callback=log_fn,
                ppo_kwargs=ppo_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            trial.set_user_attr("status", f"train_failed: {exc}")
            raise

        try:
            backtest_result = run_ppo_backtest(
                model_path=str(model_path),
                df_test=valid_df.copy(),
                initial_cash=args.initial_cash,
                monthly_invest=args.monthly_cash,
                fee=args.fee,
            )
        except Exception as exc:  # noqa: BLE001
            trial.set_user_attr("status", f"backtest_failed: {exc}")
            raise

        metrics = backtest_result.get("metrics", {}) or {}
        sharpe = float(metrics.get("夏普比率", 0.0) or 0.0)

        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("ppo_kwargs", ppo_kwargs)

        summary_payload = {
            "trial_number": trial.number,
            "params": sampled_params,
            "ppo_kwargs": ppo_kwargs,
            "metrics": metrics,
        }
        summary_path = trial_dir / "summary.json"
        try:
            with summary_path.open("w", encoding="utf-8") as fh:
                json.dump(summary_payload, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

        trial.report(sharpe, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return sharpe

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    return study


def export_best_params(study: optuna.study.Study, output_dir: Path) -> Dict[str, Any]:
    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params["value"] = best_trial.value
    if "metrics" in best_trial.user_attrs:
        best_params["metrics"] = best_trial.user_attrs["metrics"]
    if "ppo_kwargs" in best_trial.user_attrs:
        best_params["ppo_kwargs"] = best_trial.user_attrs["ppo_kwargs"]
    output_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = output_dir / "best_params.json"
    with best_params_path.open("w", encoding="utf-8") as fh:
        json.dump(best_params, fh, ensure_ascii=False, indent=2)
    return best_params


def copy_best_trial_artifacts(study: optuna.study.Study, output_dir: Path) -> Optional[Path]:
    best_trial_number = study.best_trial.number
    trial_dir = output_dir / TRIAL_SUBDIR / f"trial_{best_trial_number:04d}"
    if not trial_dir.exists():
        return None

    best_dir = output_dir / "best_trial"
    best_dir.mkdir(parents=True, exist_ok=True)

    model_src = trial_dir / MODEL_FILENAME
    if model_src.exists():
        shutil.copy2(model_src, best_dir / MODEL_FILENAME)

    scaler_src = trial_dir / SCALER_FILENAME
    if scaler_src.exists():
        shutil.copy2(scaler_src, best_dir / SCALER_FILENAME)

    summary_src = trial_dir / "summary.json"
    if summary_src.exists():
        shutil.copy2(summary_src, best_dir / "summary.json")

    return best_dir


def export_trials_dataframe(study: optuna.study.Study, output_dir: Path) -> Path:
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    csv_path = output_dir / "trials.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    data_df = load_market_dataframe(args.data_path)
    train_df, valid_df = split_train_valid(data_df, args.valid_ratio)

    print(f"训练集: {len(train_df)} 行, 验证集: {len(valid_df)} 行")
    study = run_study(args, train_df, valid_df)
    print(f"Optuna Study 完成 - 最佳 Sharpe: {study.best_value:.4f}")

    best_params = export_best_params(study, args.output_dir)
    print("最佳参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    best_dir = copy_best_trial_artifacts(study, args.output_dir)
    if best_dir is not None:
        print(f"最佳 Trial 模型与缩放器已复制至: {best_dir}")
    trials_csv = export_trials_dataframe(study, args.output_dir)
    print(f"完整 trial 历史已保存至: {trials_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
