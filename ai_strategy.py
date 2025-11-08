import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

try:  # pragma: no cover - optional dependency for RL
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:  # pragma: no cover - handled at runtime
    PPO = None
    DummyVecEnv = None

if TYPE_CHECKING:
    from stable_baselines3 import PPO as PPOType
    from stable_baselines3.common.vec_env import DummyVecEnv as DummyVecEnvType
else:  # pragma: no cover - typing fallback
    PPOType = Any  # type: ignore[misc]
    DummyVecEnvType = Any  # type: ignore[misc]

try:  # pragma: no cover - optional dependency for GPU detection
    import torch
except ImportError:  # pragma: no cover - handled at runtime
    torch = None


ProgressCallback = Callable[[float, int, Dict[str, Any]], None]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

_COLUMN_ALIASES: Dict[str, str] = {
    "open": "Open",
    "开盘": "Open",
    "openprice": "Open",
    "high": "High",
    "最高": "High",
    "low": "Low",
    "最低": "Low",
    "close": "Close",
    "收盘": "Close",
    "收盘价": "Close",
    "price": "Close",
    "volume": "Volume",
    "vol": "Volume",
    "成交量": "Volume",
    "amount": "Amount",
    "成交额": "Amount",
    "turnover": "Amount",
}


def _log(logger: Optional[logging.Logger], message: str) -> None:
    if logger is not None:
        logger.info(message)


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _COLUMN_ALIASES:
            rename_map[col] = _COLUMN_ALIASES[key]
    return df.rename(columns=rename_map)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    date_col = None
    for candidate in ("Date", "date", "交易日期", "time", "datetime"):
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        return df

    dates = pd.to_datetime(df[date_col], errors="coerce")
    df = df.loc[dates.notna()].copy()
    df.index = pd.DatetimeIndex(dates[dates.notna()])
    return df.sort_index()


def _prepare_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("训练/测试数据为空，无法继续。")

    working = _rename_columns(df.copy())
    working = _ensure_datetime_index(working)

    required = ["Open", "High", "Low", "Close"]
    for col in required:
        if col not in working.columns:
            raise ValueError(f"数据缺少必要的行情列: {col}")
        working[col] = pd.to_numeric(working[col], errors="coerce")

    if "Volume" not in working.columns:
        working["Volume"] = 0.0
    else:
        working["Volume"] = pd.to_numeric(working["Volume"], errors="coerce").fillna(0.0)

    if "Amount" in working.columns:
        working["Amount"] = pd.to_numeric(working["Amount"], errors="coerce").fillna(0.0)

    working = working.dropna(subset=["Close"])
    if working.empty:
        raise ValueError("清洗后的数据为空，请检查原始 CSV 是否存在缺失值。")

    return working


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """为数据集添加用于强化学习状态的特征。"""

    working = _prepare_price_dataframe(df)

    close = working["Close"].astype(float)
    working["returns"] = close.pct_change().fillna(0.0)
    working["ma_5"] = close.rolling(window=5, min_periods=1).mean()
    working["ma_10"] = close.rolling(window=10, min_periods=1).mean()
    working["ma_20"] = close.rolling(window=20, min_periods=1).mean()
    working["volatility_10"] = working["returns"].rolling(window=10, min_periods=1).std().fillna(0.0)
    working["momentum_10"] = close / close.shift(10) - 1.0

    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss.replace(0.0, np.nan))
    working["rsi"] = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], np.nan)))

    working.replace([np.inf, -np.inf], 0.0, inplace=True)
    working.fillna(0.0, inplace=True)
    return working


class DynamicGridEnv(gym.Env):
    """符合 Gymnasium API 的动态网格交易环境。"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_cash: float = 100_000.0,
        fee: float = 0.001,
        monthly_invest: float = 0.0,
    ) -> None:
        super().__init__()

        self.df = add_features(df.copy())
        self.initial_cash = float(initial_cash)
        self.fee = float(fee)
        self.monthly_invest = float(monthly_invest)
        self.min_commission: float = 5.0
        self.profit_bonus_coef: float = 0.5
        self.trade_penalty: float = 0.0005

        self.action_space = spaces.Box(
            low=np.array([0.005, 0.01, 0.10], dtype=np.float32),
            high=np.array([0.05, 0.10, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.current_step: int = 0
        self.cash: float = self.initial_cash
        self.holding_shares: float = 0.0
        self.portfolio_value: float = self.initial_cash
        self.trades: List[Dict[str, Any]] = []
        self.last_trade_price: float = 0.0
        self.cash_added: float = self.initial_cash
        self.total_reward: float = 0.0
        self.total_commission: float = 0.0
        self.nav_history: List[float] = []
        self.nav_index: List[Any] = []
        self._last_invest_month: Optional[Tuple[int, int]] = None
        self.current_date: Any = self._index_value(self.current_step)
        self._last_action: np.ndarray = np.zeros(self.action_space.shape, dtype=np.float32)
        self._position_cost: float = 0.0

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.holding_shares = 0.0
        self.portfolio_value = self.initial_cash
        self.trades = []
        self.last_trade_price = 0.0
        self.cash_added = self.initial_cash
        self.total_reward = 0.0
        self.total_commission = 0.0
        self._last_invest_month = None
        self.nav_history = []
        self.nav_index = []
        self.current_date = self._index_value(self.current_step)
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._position_cost = 0.0
        self._record_nav()
        return self._get_obs(), self._get_info()

    def _index_value(self, step: int) -> Any:
        if isinstance(self.df.index, pd.DatetimeIndex) and len(self.df.index) > 0:
            idx = min(max(step, 0), len(self.df.index) - 1)
            return self.df.index[idx]
        return step

    def _record_nav(self) -> None:
        self.nav_history.append(float(self.portfolio_value))
        self.nav_index.append(self._index_value(self.current_step))

    def _apply_monthly_investment(self) -> float:
        if self.monthly_invest <= 0 or not isinstance(self.df.index, pd.DatetimeIndex):
            return 0.0
        current_date = self._index_value(self.current_step)
        month_key = (int(current_date.year), int(current_date.month))
        if self._last_invest_month == month_key:
            return 0.0
        self._last_invest_month = month_key
        self.cash += self.monthly_invest
        self.cash_added += self.monthly_invest
        self.trades.append({
            "step": self.current_step,
            "type": "deposit",
            "amount": self.monthly_invest,
            "timestamp": self._index_value(self.current_step),
        })
        return self.monthly_invest

    def _get_obs(self) -> np.ndarray:
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        row = self.df.iloc[self.current_step]
        market_features = row[["ma_5", "ma_10", "ma_20", "volatility_10", "momentum_10", "rsi"]]
        market_features = market_features.astype(float)
        mean = market_features.mean()
        std = market_features.std()
        market_features = (market_features - mean) / (std + 1e-8)
        market_features = np.nan_to_num(market_features, nan=0.0)
        market_features = np.asarray(market_features, dtype=np.float32)

        unrealized_pnl = 0.0
        if self.holding_shares > 0 and self.last_trade_price > 0:
            unrealized_pnl = (float(row["Close"]) - self.last_trade_price) / max(self.last_trade_price, 1e-8)

        account_state = np.array(
            [
                self.portfolio_value / max(self.cash_added, 1e-8),
                self.cash / max(self.portfolio_value, 1e-8),
                (self.holding_shares * float(row["Close"])) / max(self.portfolio_value, 1e-8),
                unrealized_pnl,
            ],
            dtype=np.float32,
        )

        return np.concatenate([market_features, account_state])

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self.current_step,
            "portfolio_value": float(self.portfolio_value),
            "cash": float(self.cash),
            "holding_shares": float(self.holding_shares),
            "cash_added": float(self.cash_added),
            "current_date": self._index_value(self.current_step),
            "trades": [],
            "last_action": self._last_action.tolist(),
        }

    def step(self, action: Sequence[float]):
        if self.current_step >= len(self.df) - 1:
            info_terminal = self._get_info()
            info_terminal["trades"] = []
            return self._get_obs(), 0.0, True, False, info_terminal

        action_array = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        self._last_action = action_array.copy()
        position_size = float(action_array[2])
        current_row = self.df.iloc[self.current_step]
        current_price = float(current_row["Close"])
        self.current_date = self._index_value(self.current_step)
        trades_this_step: List[Dict[str, Any]] = []
        deposit = self._apply_monthly_investment()
        prev_value = max(self.portfolio_value, 1e-8)

        target_value = self.portfolio_value * position_size
        current_value = self.holding_shares * current_price

        if current_price <= 0:
            current_price = max(current_price, 1e-6)

        if target_value > current_value + 1e-8:
            amount_needed = target_value - current_value
            max_affordable = self._max_affordable_buy_amount()
            amount_to_buy = min(max(amount_needed, 0.0), max_affordable)
            if amount_to_buy > 0:
                shares_to_buy = amount_to_buy / current_price
                commission_paid = self._calculate_commission(amount_to_buy)
                total_cost = amount_to_buy + commission_paid
                self.holding_shares += shares_to_buy
                self.cash -= total_cost
                self.last_trade_price = current_price
                self.total_commission += commission_paid
                self._position_cost += total_cost
                trade_entry = {
                    "step": self.current_step,
                    "type": "buy",
                    "shares": float(shares_to_buy),
                    "price": current_price,
                    "value": float(amount_to_buy),
                    "timestamp": self.current_date,
                    "commission": float(commission_paid),
                    "profit": float(-total_cost),
                }
                self.trades.append(trade_entry)
                trades_this_step.append(
                    {
                        "时间": self.current_date,
                        "方向": "买入",
                        "价格": current_price,
                        "数量": float(shares_to_buy),
                        "金额": -float(total_cost),
                        "手续费": float(commission_paid),
                        "盈亏": 0.0,
                    }
                )
        elif target_value + 1e-8 < current_value and self.holding_shares > 0:
            amount_to_release = current_value - target_value
            shares_to_sell = min(self.holding_shares, amount_to_release / current_price)
            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price
                commission_paid = self._calculate_commission(proceeds)
                holding_before = max(self.holding_shares, 1e-8)
                ratio = float(shares_to_sell) / holding_before
                cost_released = self._position_cost * ratio
                net_proceeds = proceeds - commission_paid
                self.holding_shares -= shares_to_sell
                self.cash += net_proceeds
                self.total_commission += commission_paid
                self._position_cost = max(self._position_cost - cost_released, 0.0)
                profit = net_proceeds - cost_released
                trade_entry = {
                    "step": self.current_step,
                    "type": "sell",
                    "shares": float(shares_to_sell),
                    "price": current_price,
                    "value": float(proceeds),
                    "timestamp": self.current_date,
                    "commission": float(commission_paid),
                    "profit": float(profit),
                }
                self.trades.append(trade_entry)
                trades_this_step.append(
                    {
                        "时间": self.current_date,
                        "方向": "卖出",
                        "价格": current_price,
                        "数量": float(shares_to_sell),
                        "金额": float(net_proceeds),
                        "手续费": float(commission_paid),
                        "盈亏": float(profit),
                    }
                )

        self.portfolio_value = self.cash + self.holding_shares * current_price

        base_reward = (self.portfolio_value - prev_value - deposit) / max(prev_value, 1.0)

        profit_bonus = 0.0
        if trades_this_step:
            positive_profit = sum(
                float(trade.get("盈亏", 0.0))
                for trade in trades_this_step
                if str(trade.get("方向")) == "卖出" and float(trade.get("盈亏", 0.0)) > 0.0
            )
            if positive_profit > 0.0:
                profit_bonus = (positive_profit / max(self.portfolio_value, 1.0)) * self.profit_bonus_coef

        trade_penalty = self.trade_penalty if trades_this_step else 0.0

        reward = base_reward + profit_bonus - trade_penalty

        self.total_reward += reward

        self.current_step += 1
        terminated = self.portfolio_value <= self.initial_cash * 0.3
        truncated = self.current_step >= len(self.df) - 1

        self._record_nav()
        info = self._get_info()
        info["trades"] = trades_this_step
        info["last_action"] = action_array.tolist()
        return self._get_obs(), float(reward), terminated, truncated, info

    def _max_affordable_buy_amount(self) -> float:
        cash_available = max(self.cash, 0.0)
        if cash_available <= 0 or self.fee < 0:
            return 0.0
        min_fee = max(self.min_commission, 0.0)
        if self.fee == 0:
            return max(cash_available - min_fee, 0.0) if min_fee > 0 else cash_available
        if cash_available <= min_fee:
            return 0.0
        threshold = min_fee / self.fee if self.fee > 0 else float("inf")
        candidate = cash_available - min_fee
        if candidate < threshold:
            return max(candidate, 0.0)
        return cash_available / (1.0 + self.fee)

    def _calculate_commission(self, trade_amount: float) -> float:
        if trade_amount <= 0 or self.fee <= 0:
            return 0.0
        return max(trade_amount * self.fee, self.min_commission)

    def get_nav_series(self) -> pd.Series:
        index = self.nav_index
        if isinstance(self.df.index, pd.DatetimeIndex):
            index = pd.DatetimeIndex(index)
        return pd.Series(self.nav_history, index=index, name="AI策略")

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            print(
                f"Step: {self.current_step}, NAV: {self.portfolio_value:.2f}, Cash: {self.cash:.2f}, "
                f"Shares: {self.holding_shares:.4f}"
            )

    def close(self) -> None:  # pragma: no cover - nothing to clean up
        pass


@dataclass
class TrainingConfig:
    epochs: int = 20
    total_timesteps: int = 200_000
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    stop_loss_init: float = 0.05
    take_profit_init: float = 0.10
    monthly_cash: float = 0.0
    fee: float = 0.001
    device_preference: str = "auto"
    seed: int = 42
    n_steps: int = 1024
    eval_episodes: int = 1
    logger: Optional[logging.Logger] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("logger", None)
        return data


@dataclass
class TrainingArtifact:
    framework: str
    model_path: str
    config_path: str
    metadata_path: str
    epochs: int
    total_timesteps: int
    monthly_cash: float
    fee: float
    device_preference: str
    best_epoch: Optional[int] = None
    best_metrics: Dict[str, Any] = field(default_factory=dict)
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _utc_now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "metadata_path": self.metadata_path,
            "epochs": self.epochs,
            "total_timesteps": self.total_timesteps,
            "monthly_cash": self.monthly_cash,
            "fee": self.fee,
            "device_preference": self.device_preference,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "training_history": self.training_history,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingArtifact":
        return cls(
            framework=data["framework"],
            model_path=data["model_path"],
            config_path=data.get("config_path", ""),
            metadata_path=data.get("metadata_path", ""),
            epochs=data.get("epochs", 0),
            total_timesteps=data.get("total_timesteps", 0),
            monthly_cash=data.get("monthly_cash", 0.0),
            fee=data.get("fee", 0.0),
            device_preference=data.get("device_preference", "auto"),
            best_epoch=data.get("best_epoch"),
            best_metrics=data.get("best_metrics", {}),
            training_history=data.get("training_history", []),
            created_at=data.get("created_at", _utc_now().isoformat()),
        )


def _resolve_device(device_pref: str) -> str:
    pref = (device_pref or "auto").lower()
    if torch is None:
        return "cpu"
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return pref


def _make_vec_env(df: pd.DataFrame, config: TrainingConfig, initial_cash: float) -> DummyVecEnvType:
    if DummyVecEnv is None:
        raise ImportError(
            "缺少 stable-baselines3 库，请先执行 `pip install stable-baselines3[extra]` 后重试。"
        )

    def _init_env() -> DynamicGridEnv:
        env = DynamicGridEnv(
            df=df,
            initial_cash=initial_cash,
            fee=config.fee,
            monthly_invest=config.monthly_cash,
        )
        return env

    return DummyVecEnv([_init_env])


def _run_single_episode(
    model: PPOType,
    df: pd.DataFrame,
    initial_cash: float,
    fee: float,
    monthly_cash: float,
) -> Tuple[Dict[str, Any], pd.Series]:
    env = DynamicGridEnv(df=df, initial_cash=initial_cash, fee=fee, monthly_invest=monthly_cash)
    obs, _ = env.reset()
    total_reward = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    metrics = {
        "final_nav": float(env.portfolio_value),
        "cash_added": float(env.cash_added),
        "num_trades": sum(1 for t in env.trades if t.get("type") in {"buy", "sell"}),
        "total_reward": float(total_reward),
    }
    return metrics, env.get_nav_series()


def _evaluate_policy(
    model: PPOType,
    df: pd.DataFrame,
    initial_cash: float,
    fee: float,
    monthly_cash: float,
    episodes: int,
) -> Dict[str, Any]:
    episode_metrics: List[Dict[str, Any]] = []
    history_series: Optional[pd.Series] = None
    for _ in range(max(1, episodes)):
        metrics, nav_series = _run_single_episode(model, df, initial_cash, fee, monthly_cash)
        episode_metrics.append(metrics)
        if history_series is None:
            history_series = nav_series

    avg = {
        key: float(np.mean([m[key] for m in episode_metrics]))
        for key in episode_metrics[0]
    }
    avg["equity_curve"] = history_series
    return avg


def auto_calibrate_training_config(
    df: pd.DataFrame,
    monthly_cash: float = 0.0,
    fee: float = 0.001,
    benchmark_close: Optional[pd.Series] = None,
) -> TrainingConfig:
    working = _prepare_price_dataframe(df)
    num_rows = len(working)
    epochs = int(np.clip(num_rows // 180, 5, 60))
    volatility = float(working["Close"].pct_change().std() or 0.0)
    total_timesteps = max(int(epochs * 4000), num_rows * 20)
    batch_size = 128 if num_rows > 2500 else 64
    learning_rate = 3e-4 if num_rows > 1000 else 5e-4
    stop_loss = float(np.clip(volatility * 3.0, 0.02, 0.08))
    take_profit = float(np.clip(volatility * 6.0, 0.05, 0.20))

    if benchmark_close is not None and not benchmark_close.empty:
        benchmark_vol = benchmark_close.pct_change().std()
        if benchmark_vol and not np.isnan(benchmark_vol):
            stop_loss = float(np.clip(benchmark_vol * 2.5, 0.02, 0.08))
            take_profit = float(np.clip(benchmark_vol * 5.0, 0.05, 0.20))

    config = TrainingConfig(
        epochs=epochs,
        total_timesteps=total_timesteps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        stop_loss_init=stop_loss,
        take_profit_init=take_profit,
        monthly_cash=monthly_cash,
        fee=fee,
    )
    return config


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def train_ai_strategy(
    df: pd.DataFrame,
    framework: str,
    config: TrainingConfig,
    output_dir: Path,
    benchmark_close: Optional[pd.Series] = None,
    verbose: bool = False,
    initial_cash: float = 100_000.0,
    progress_callback: Optional[ProgressCallback] = None,
) -> TrainingArtifact:
    if PPO is None or DummyVecEnv is None:
        raise ImportError(
            "未检测到 stable-baselines3，请先执行 `pip install stable-baselines3[extra]` 安装依赖。"
        )

    logger = config.logger if verbose else config.logger
    train_df = _prepare_price_dataframe(df)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(config.device_preference)
    _log(logger, f"使用设备: {device}")

    if torch is not None:
        torch.manual_seed(config.seed)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    np.random.seed(config.seed)

    vec_env = _make_vec_env(train_df, config, initial_cash)
    try:
        vec_env.reset(seed=config.seed)
    except TypeError:
        # stable-baselines3 < 2.0 does not support the seed keyword
        vec_env.seed(config.seed)  # type: ignore[attr-defined]
        vec_env.reset()

    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        n_steps=min(config.n_steps, max(128, len(train_df) // 2)),
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        verbose=0,
        policy_kwargs=policy_kwargs,
        device=device,
    )

    total_timesteps = max(config.total_timesteps, config.epochs)
    epochs = max(config.epochs, 1)
    base_per_epoch = total_timesteps // epochs
    remainder = total_timesteps % epochs

    best_metrics: Optional[Dict[str, Any]] = None
    best_epoch: Optional[int] = None
    history: List[Dict[str, Any]] = []
    model_path = output_dir / "ai_pytorch_model.zip"
    config_path = output_dir / "training_config.json"
    metadata_path = output_dir / "training_artifact.json"

    timesteps_run = 0
    for epoch_idx in range(1, epochs + 1):
        extra = 1 if epoch_idx <= remainder else 0
        run_steps = base_per_epoch + extra
        timesteps_run += run_steps

        _log(logger, f"Epoch {epoch_idx}/{epochs}: 训练 {run_steps} timesteps ...")
        model.learn(total_timesteps=run_steps, reset_num_timesteps=False, progress_bar=False)

        eval_stats = _evaluate_policy(
            model,
            train_df,
            initial_cash=initial_cash,
            fee=config.fee,
            monthly_cash=config.monthly_cash,
            episodes=config.eval_episodes,
        )

        history_entry = {
            "epoch": epoch_idx,
            "timesteps": timesteps_run,
            "final_nav": float(eval_stats["final_nav"]),
            "cash_added": float(eval_stats["cash_added"]),
            "reward": float(eval_stats["total_reward"]),
            "num_trades": int(eval_stats["num_trades"]),
            "timestamp": _utc_now().isoformat(),
        }
        history.append(history_entry)

        improved = False
        if best_metrics is None or eval_stats["final_nav"] > best_metrics.get("final_nav", -np.inf):
            best_metrics = {
                "final_nav": float(eval_stats["final_nav"]),
                "cash_added": float(eval_stats["cash_added"]),
                "total_reward": float(eval_stats["total_reward"]),
                "num_trades": int(eval_stats["num_trades"]),
            }
            best_epoch = epoch_idx
            improved = True
            model.save(str(model_path))
            _log(logger, f"更新最佳模型: Epoch {epoch_idx}, 净值 {best_metrics['final_nav']:.2f}")

        if progress_callback is not None:
            progress_callback(
                float(epoch_idx),
                epochs,
                {
                    "progress_ratio": epoch_idx / epochs,
                    "epoch_progress": epoch_idx,
                    "final_nav": float(eval_stats["final_nav"]),
                    "cash_added": float(eval_stats["cash_added"]),
                    "reward": float(eval_stats["total_reward"]),
                    "best_final_nav": best_metrics.get("final_nav") if best_metrics else None,
                    "best_cash_added": best_metrics.get("cash_added") if best_metrics else None,
                    "best_epoch_so_far": best_epoch,
                    "model_improved": improved,
                },
            )

    if best_metrics is None:
        model.save(str(model_path))

    vec_env.close()

    _save_json(config_path, config.to_dict())
    artifact = TrainingArtifact(
        framework=framework,
        model_path=str(model_path),
        config_path=str(config_path),
        metadata_path=str(metadata_path),
        epochs=epochs,
        total_timesteps=total_timesteps,
        monthly_cash=config.monthly_cash,
        fee=config.fee,
        device_preference=config.device_preference,
        best_epoch=best_epoch,
        best_metrics=best_metrics or {},
        training_history=history,
    )
    _save_json(metadata_path, artifact.to_dict())
    return artifact


def load_artifact(path: Path | str) -> TrainingArtifact:
    path = Path(path)
    meta_path = path if path.is_file() else path / "training_artifact.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"找不到训练产物元数据文件: {meta_path}")
    data = _load_json(meta_path)
    artifact = TrainingArtifact.from_dict(data)
    if not Path(artifact.model_path).exists():
        raise FileNotFoundError(f"找不到训练好的模型文件: {artifact.model_path}")
    return artifact


def _simulate_dca(df: pd.DataFrame, initial_cash: float, monthly_invest: float, fee: float) -> Tuple[pd.Series, float]:
    cash = float(initial_cash)
    invested = float(initial_cash)
    shares = 0.0
    last_month: Optional[Tuple[int, int]] = None
    equity_values: List[float] = []

    for idx, (date, row) in enumerate(df.iterrows()):
        price = float(row["Close"])
        if price <= 0:
            price = np.nan
        if np.isnan(price):
            equity_values.append(cash + shares * (equity_values[-1] if equity_values else 0.0))
            continue

        month_key = None
        if isinstance(df.index, pd.DatetimeIndex):
            month_key = (date.year, date.month)

        if idx == 0:
            amount = cash
            shares += (amount * (1 - fee)) / price
            cash = 0.0
        else:
            if monthly_invest > 0 and month_key is not None and month_key != last_month:
                cash += monthly_invest
                invested += monthly_invest
                amount = cash
                if amount > 0:
                    shares += (amount * (1 - fee)) / price
                    cash = 0.0
        last_month = month_key
        equity_values.append(cash + shares * price)

    equity_series = pd.Series(equity_values, index=df.index, name="定投策略")
    return equity_series, invested


def _simulate_buy_and_hold(df: pd.DataFrame, initial_cash: float, fee: float) -> pd.Series:
    first_price = float(df["Close"].iloc[0])
    shares = (initial_cash * (1 - fee)) / max(first_price, 1e-8)
    equity = shares * df["Close"].astype(float)
    return equity.rename("买入持有")


def _compute_drawdown(series: pd.Series) -> pd.Series:
    cummax = series.cummax()
    drawdown = series / cummax.replace(0, np.nan) - 1.0
    return drawdown.fillna(0.0)


def _compute_performance_metrics(equity: pd.Series, invested: float) -> Dict[str, float]:
    final_value = float(equity.iloc[-1])
    total_return = (final_value / invested - 1.0) if invested > 0 else 0.0
    if isinstance(equity.index, pd.DatetimeIndex) and len(equity.index) > 1:
        days = max((equity.index[-1] - equity.index[0]).days, 1)
    else:
        days = max(len(equity), 1)
    annualized = ((1 + total_return) ** (365 / days) - 1) if days > 1 else total_return
    drawdown = abs(float(_compute_drawdown(equity).min()))
    daily_returns = equity.pct_change().dropna()
    sharpe = 0.0
    if not daily_returns.empty and daily_returns.std() > 0:
        sharpe = float((daily_returns.mean() / daily_returns.std()) * math.sqrt(252))
    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "max_drawdown": float(drawdown),
        "sharpe": float(sharpe),
    }


def _normalize_benchmark_series(
    benchmark_close: pd.Series,
    target_index: pd.Index,
    initial_cash: float,
) -> pd.Series:
    series = benchmark_close.copy().astype(float)
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index, errors="coerce")
    series = series.loc[series.index.notna()].sort_index()
    series = series.reindex(target_index).fillna(method="ffill").dropna()
    if series.empty:
        raise ValueError("基准数据与测试区间没有重叠的日期。")
    normalized = initial_cash * (series / series.iloc[0])
    return normalized.rename("基准指数")


def run_ai_comparison_backtest(
    df: pd.DataFrame,
    artifact: TrainingArtifact,
    initial_cash: float,
    monthly_investment_amount: float,
    fee: float,
    benchmark_df: Optional[pd.DataFrame] = None,
    benchmark_close: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    test_df = _prepare_price_dataframe(df)

    model = PPO.load(artifact.model_path, device=_resolve_device("cpu"))

    ai_metrics, ai_equity = _run_single_episode(
        model,
        test_df,
        initial_cash=initial_cash,
        fee=fee,
        monthly_cash=monthly_investment_amount,
    )

    dca_equity, dca_invested = _simulate_dca(test_df, initial_cash, monthly_investment_amount, fee)
    buy_hold_equity = _simulate_buy_and_hold(test_df, initial_cash, fee)

    equity_frames = {
        "AI策略": ai_equity,
        "定投策略": dca_equity,
        "买入持有": buy_hold_equity,
    }

    if benchmark_close is None and benchmark_df is not None and "Close" in benchmark_df.columns:
        benchmark_close = benchmark_df.set_index("Date")["Close"] if "Date" in benchmark_df.columns else benchmark_df["Close"]

    if benchmark_close is not None and not benchmark_close.empty:
        try:
            benchmark_equity = _normalize_benchmark_series(benchmark_close, test_df.index, initial_cash)
            equity_frames["基准指数"] = benchmark_equity
        except Exception:
            pass

    equity_df = pd.concat(equity_frames.values(), axis=1).ffill().dropna(how="all")
    drawdown = {col: _compute_drawdown(equity_df[col]) for col in equity_df.columns}

    metrics = {
        "ai": _compute_performance_metrics(equity_df["AI策略"], ai_metrics["cash_added"]),
        "dca": _compute_performance_metrics(equity_df["定投策略"], dca_invested),
        "buy_and_hold": _compute_performance_metrics(equity_df["买入持有"], initial_cash),
    }
    if "基准指数" in equity_df.columns:
        metrics["benchmark"] = _compute_performance_metrics(equity_df["基准指数"], initial_cash)

    result = {
        "equity": equity_df,
        "drawdown": drawdown,
        "metrics": metrics,
        "ai_episode": ai_metrics,
        "dca_details": {"total_invested": dca_invested},
        "artifact": artifact.to_dict(),
    }
    return result


__all__ = [
    "DynamicGridEnv",
    "TrainingConfig",
    "TrainingArtifact",
    "add_features",
    "auto_calibrate_training_config",
    "train_ai_strategy",
    "load_artifact",
    "run_ai_comparison_backtest",
]
