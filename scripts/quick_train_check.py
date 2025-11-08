"""Smoke test for the PPO training pipeline.

This script runs a tiny training session to ensure the RL pipeline works
without raising warnings or runtime errors. Warnings are promoted to
exceptions so we can catch regressions early.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_strategy import TrainingConfig, train_ai_strategy


def _resolve_data_path() -> Path:
    candidates = [
        Path("data/stock_data.csv"),
        Path("data/sh510300.csv"),
        Path("data/sh000001.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("未找到用于训练的数据文件，请将 CSV 放在 data/ 目录下。")


def run_smoke_test() -> None:
    warnings.filterwarnings("error")

    data_path = _resolve_data_path()
    df = pd.read_csv(data_path)

    config = TrainingConfig(
        epochs=1,
        total_timesteps=512,
        batch_size=64,
        learning_rate=3e-4,
        n_steps=128,
        eval_episodes=1,
        device_preference="cpu",
        fee=0.001,
        monthly_cash=0.0,
    )

    output_dir = Path("checkpoint/ai_strategy/quick_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = train_ai_strategy(
        df=df,
        framework="pytorch",
        config=config,
        output_dir=output_dir,
        benchmark_close=None,
        verbose=False,
        initial_cash=100_000.0,
    )

    print("✅ Smoke training finished successfully:", artifact.model_path)


if __name__ == "__main__":
    run_smoke_test()
