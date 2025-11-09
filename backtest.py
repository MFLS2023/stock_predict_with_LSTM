from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
except ImportError as exc:
    raise ImportError(
        "依赖缺失：无法导入 'backtesting' 库，请先执行 `pip install backtesting` (或安装 requirements.txt) 后再运行 GUI。"
    ) from exc

try:  # pragma: no cover - optional PPO 依赖
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover - 延迟到调用处提示
    PPO = None  # type: ignore

from ai_strategy import DynamicGridEnv

def run_backtest(df: pd.DataFrame, initial_cash: float = 100_000.0, fee: float = 0.001):
    """
    运行基于“智行”规则的移动平均线交叉策略回测。
    这是 v2.0 方案中的基准策略之一。
    """

    class ZhixingMaStrategy(Strategy):
        # 定义两条移动平均线的周期
        n1 = 10
        n2 = 20

        def init(self):
            # 计算移动平均线
            self.ma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), self.data.Close)
            self.ma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), self.data.Close)

        def next(self):
            # 如果快线（MA10）上穿慢线（MA20），并且当前没有持仓，则买入
            if crossover(self.ma1, self.ma2) and not self.position:
                self.buy()
            # 如果快线下穿慢线，并且当前有持仓，则卖出
            elif crossover(self.ma2, self.ma1) and self.position:
                self.position.close()

    # 格式化数据以符合 backtesting.py 的要求
    # 列名必须是 'Open', 'High', 'Low', 'Close', 'Volume'
    bt_df = df.copy()
    bt_df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    }, inplace=True)
    bt_df = bt_df.set_index('Date')

    commission = max(0.0, min(float(fee), 1.0))

    bt = Backtest(bt_df, ZhixingMaStrategy, cash=initial_cash, commission=commission)
    stats = bt.run()

    equity_curve = stats['_equity_curve']['Equity']
    equity_curve.name = "智行均线策略"

    trade_log: List[Dict[str, Any]] = []
    trades_df_raw = stats.get('_trades')
    trades_df = pd.DataFrame(trades_df_raw) if trades_df_raw is not None else pd.DataFrame()
    if not trades_df.empty:
        trades_df = trades_df.reset_index(drop=True)
        for _, row in trades_df.iterrows():
            exit_time = row.get('ExitTime') or row.get('EntryTime')
            exit_price = row.get('ExitPrice', row.get('Price', row.get('EntryPrice', 0.0)))
            size = float(abs(row.get('Size', 0.0)))
            pnl = float(row.get('PnL', 0.0))
            commission = float(row.get('EntryCommission', 0.0) or 0.0) + float(row.get('ExitCommission', 0.0) or 0.0)
            trade_log.append(
                {
                    "时间": exit_time,
                    "方向": "卖出",
                    "价格": float(exit_price),
                    "数量": size,
                    "金额": pnl,
                    "手续费": commission,
                    "盈亏": pnl,
                }
            )

    metrics = _calculate_metrics(equity_curve, trade_log)
    metrics.update(
        {
            "Total Return": stats['Return [%]'] / 100,
            "Annualized Return": stats['Return (Ann.) [%]'] / 100,
            "Max Drawdown (BT)": -stats['Max. Drawdown [%]'] / 100,
            "Sharpe Ratio (BT)": stats['Sharpe Ratio'],
            "Win Rate (BT)": stats['Win Rate [%]'] / 100,
            "Total Trades (BT)": stats['# Trades'],
            "Final Equity (BT)": stats['Equity Final [$]'],
        }
    )

    return {
        "equity_curve": equity_curve,
        "metrics": metrics,
        "trades_df": pd.DataFrame(trade_log),
        "actions_df": pd.DataFrame(),
        "raw": {
            "stats": stats,
            "trades": trades_df_raw,
        },
    }

def run_grid_backtest(*args, **kwargs):
    """网格策略回测的占位符函数。"""
    print("警告: run_grid_backtest() 尚未在此次恢复中实现。")
    raise NotImplementedError("网格策略回测功能待实现。")

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "Date" in df.columns:
        dated = pd.to_datetime(df["Date"], errors="coerce")
        frame = df.loc[dated.notna()].copy()
        frame.index = pd.DatetimeIndex(dated[dated.notna()])
        return frame.drop(columns=["Date"], errors="ignore").sort_index()
    return df.copy()


def _calculate_metrics(equity_curve: pd.Series, trade_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    cleaned_curve = equity_curve.dropna()
    if cleaned_curve.empty:
        return {
            "总回报率": 0.0,
            "年化回报率": 0.0,
            "最大回撤": 0.0,
            "夏普比率": 0.0,
            "胜率": 0.0,
            "总交易次数": len(trade_log),
            "最终净值": 0.0,
        }

    start_value = float(cleaned_curve.iloc[0])
    end_value = float(cleaned_curve.iloc[-1])
    total_return = (end_value / max(start_value, 1e-8)) - 1.0

    rolling_max = cleaned_curve.cummax()
    drawdowns = cleaned_curve / rolling_max.replace(0, np.nan) - 1.0
    max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0

    if isinstance(cleaned_curve.index, pd.DatetimeIndex) and len(cleaned_curve) > 1:
        days = max((cleaned_curve.index[-1] - cleaned_curve.index[0]).days, 1)
        annualized = (1 + total_return) ** (365 / days) - 1 if days > 0 else total_return
    else:
        periods = max(len(cleaned_curve) - 1, 1)
        annualized = (1 + total_return) ** (252 / periods) - 1 if periods > 0 else total_return

    daily_returns = cleaned_curve.pct_change().dropna()
    if not daily_returns.empty and daily_returns.std() > 0:
        sharpe_ratio = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
    else:
        sharpe_ratio = 0.0

    sell_trades = [t for t in trade_log if str(t.get("方向")) == "卖出"]
    winning_sells = sum(1 for t in sell_trades if float(t.get("盈亏", t.get("金额", 0.0))) > 0)
    win_rate = winning_sells / len(sell_trades) if sell_trades else 0.0

    return {
        "总回报率": float(total_return),
        "年化回报率": float(annualized),
        "最大回撤": float(max_drawdown),
        "夏普比率": float(sharpe_ratio),
        "胜率": float(win_rate),
        "总交易次数": len(trade_log),
        "最终净值": end_value,
    }


def _compute_composite_scores(strategies: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    metric_config = {
        "总回报率": {"weight": 0.5, "higher": True},
        "夏普比率": {"weight": 0.3, "higher": True},
        "最大回撤": {"weight": 0.2, "higher": False},
    }

    collected: Dict[str, List[float]] = {key: [] for key in metric_config}
    for metrics in (data.get("metrics", {}) for data in strategies.values()):
        for name in metric_config:
            value = metrics.get(name)
            if value is not None and isinstance(value, (int, float)) and math.isfinite(float(value)):
                collected[name].append(float(value))

    scores: Dict[str, float] = {}
    for strategy_name, data in strategies.items():
        metrics = data.get("metrics", {})
        score = 0.0
        total_weight = 0.0
        for metric_name, cfg in metric_config.items():
            weight = float(cfg["weight"])
            total_weight += weight
            value = metrics.get(metric_name)
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                normalized = 0.5
            else:
                values = collected.get(metric_name, [])
                if not values:
                    normalized = 0.5
                else:
                    max_v = max(values)
                    min_v = min(values)
                    if math.isclose(max_v, min_v):
                        normalized = 0.5
                    else:
                        normalized = (float(value) - min_v) / (max_v - min_v)
                if not cfg["higher"]:
                    normalized = 1.0 - normalized
                normalized = max(0.0, min(1.0, normalized))
            score += normalized * weight
        scores[strategy_name] = score / total_weight if total_weight > 0 else 0.0
    return scores


def run_ppo_backtest(
    model_path: str,
    df_test: pd.DataFrame,
    initial_cash: float,
    *,
    monthly_invest: float = 0.0,
    fee: float = 0.001,
) -> Dict[str, Any]:
    """执行 PPO 策略回测并返回完整分析所需的数据。"""

    if PPO is None:  # pragma: no cover - 提示用户安装
        raise ImportError(
            "依赖缺失：无法导入 stable_baselines3.PPO，请先安装 `pip install stable-baselines3`."
        )

    if not model_path:
        raise ValueError("model_path 不能为空。")

    if df_test is None or df_test.empty:
        raise ValueError("测试数据为空，无法进行 PPO 回测。")

    model_path_obj = Path(model_path).expanduser()
    candidate_paths = [model_path_obj]
    if model_path_obj.suffix.lower() != ".zip":
        candidate_paths.append(model_path_obj.with_suffix((model_path_obj.suffix or "") + ".zip"))
    else:
        candidate_paths.append(Path(f"{model_path_obj}.zip"))

    resolved_path: Optional[Path] = next((p for p in candidate_paths if p.exists()), None)
    if resolved_path is None:
        tried = "\n".join(f" - {p}" for p in candidate_paths)
        raise FileNotFoundError(
            "PPO 模型文件不存在，请确认路径是否正确，已尝试:\n" + tried
        )

    print(f"开始执行PPO回测，模型: {resolved_path}")

    df_prepared = _ensure_datetime_index(df_test)
    feature_scaler = None
    scaler_candidates = [
        resolved_path.with_name("feature_scaler.pkl"),
        resolved_path.parent / "feature_scaler.pkl",
    ]
    for candidate in scaler_candidates:
        if candidate.is_file():
            try:
                with candidate.open("rb") as fh:
                    feature_scaler = pickle.load(fh)
                print(f"加载特征标准化器: {candidate}")
                break
            except Exception as exc:
                print(f"警告: 无法加载特征标准化器 {candidate}: {exc}")

    model = PPO.load(str(resolved_path))
    env = DynamicGridEnv(
        df=df_prepared,
        initial_cash=float(initial_cash),
        monthly_invest=float(monthly_invest),
        fee=float(fee),
        feature_scaler=feature_scaler,
        fit_feature_scaler=feature_scaler is None,
    )

    obs, info = env.reset()
    terminated = False
    truncated = False
    trade_log: List[Dict[str, Any]] = []
    action_log: List[List[float]] = []

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        action_log.append(list(info.get("last_action", action.tolist())))
        step_trades = info.get("trades") or []
        trade_log.extend(step_trades)

    equity_curve = env.get_nav_series()
    equity_curve.name = "PPO 动态策略"

    action_columns = ["网格间距", "止损比例", "仓位比例"]
    action_count = len(action_log)
    if action_count > 0:
        if len(equity_curve) > 1:
            action_index = equity_curve.index[1 : min(len(equity_curve), action_count + 1)]
        else:
            action_index = pd.RangeIndex(start=0, stop=action_count)
        if len(action_index) != action_count:
            action_index = pd.RangeIndex(start=0, stop=action_count)
        actions_df = pd.DataFrame(action_log, columns=action_columns, index=action_index)
    else:
        actions_df = pd.DataFrame(columns=action_columns)

    trades_df = pd.DataFrame(trade_log)
    if not trades_df.empty and "时间" in trades_df.columns:
        trades_df["时间"] = pd.to_datetime(trades_df["时间"], errors="ignore")

    metrics = _calculate_metrics(equity_curve, trade_log)
    metrics.update(
        {
            "总收益奖励": float(env.total_reward),
            "投入资金": float(env.cash_added),
        }
    )

    return {
        "equity_curve": equity_curve,
        "metrics": metrics,
        "actions_df": actions_df,
        "trades_df": trades_df,
        "raw": {
            "nav_history": env.nav_history,
            "trades": env.trades,
            "cash_added": env.cash_added,
        },
    }


def run_dca_backtest(
    df_test: pd.DataFrame,
    initial_cash: float,
    monthly_investment: float,
    *,
    fee: float = 0.001,
) -> Dict[str, Any]:
    """按月定投策略回测，返回统一格式的数据结构。"""

    df_prepared = _ensure_datetime_index(df_test)
    cash = float(initial_cash)
    shares_held = 0.0
    equity_values: List[float] = []
    equity_index: List[Any] = []
    trade_log: List[Dict[str, Any]] = []
    total_invested = float(initial_cash)
    last_month: Optional[tuple[int, int]] = None

    for date, row in df_prepared.iterrows():
        price = float(row.get("Close", 0.0))
        if not np.isfinite(price) or price <= 0:
            equity_values.append(cash + shares_held * max(price, 0.0))
            equity_index.append(date)
            continue

        month_key: Optional[tuple[int, int]] = None
        if isinstance(df_prepared.index, pd.DatetimeIndex):
            month_key = (int(date.year), int(date.month))  # type: ignore[union-attr]

        if month_key is None or month_key != last_month:
            if month_key is not None and monthly_investment > 0:
                cash += monthly_investment
                total_invested += monthly_investment
        last_month = month_key

        if cash > 0:
            commission = cash * fee
            shares_bought = (cash - commission) / price
            if shares_bought > 0:
                trade_log.append(
                    {
                        "时间": date,
                        "方向": "买入",
                        "价格": price,
                        "数量": float(shares_bought),
                        "金额": -float(cash),
                        "手续费": float(commission),
                        "盈亏": 0.0,
                    }
                )
                shares_held += shares_bought
                cash = 0.0

        equity_values.append(cash + shares_held * price)
        equity_index.append(date)

    equity_curve = pd.Series(equity_values, index=equity_index, name="定投策略")
    trades_df = pd.DataFrame(trade_log)
    if not trades_df.empty and "时间" in trades_df.columns:
        trades_df["时间"] = pd.to_datetime(trades_df["时间"], errors="ignore")

    metrics = _calculate_metrics(equity_curve, trade_log)
    metrics.update({"总投入金额": total_invested})

    return {
        "equity_curve": equity_curve,
        "metrics": metrics,
        "trades_df": trades_df,
        "actions_df": pd.DataFrame(),
        "raw": {
            "total_invested": total_invested,
        },
    }


def run_ai_comparison_backtest(
    model_path: str,
    df_test: pd.DataFrame,
    *,
    initial_cash: float = 100_000.0,
    monthly_investment: float = 0.0,
    fee: float = 0.001,
) -> Dict[str, Any]:
    """调度多策略回测并返回综合对比结果。"""

    print("开始执行策略对比回测...")

    df_prepared = _ensure_datetime_index(df_test)

    ppo_result = run_ppo_backtest(
        model_path=model_path,
        df_test=df_prepared,
        initial_cash=initial_cash,
        monthly_invest=monthly_investment,
        fee=fee,
    )

    baseline_input = (
        df_prepared.reset_index().rename(columns={"index": "Date"})
        if isinstance(df_prepared.index, pd.DatetimeIndex)
        else df_prepared.copy()
    )
    zhixing_result = run_backtest(baseline_input, initial_cash=initial_cash, fee=fee)

    dca_result = run_dca_backtest(
        df_test=df_prepared,
        initial_cash=initial_cash,
        monthly_investment=monthly_investment,
        fee=fee,
    )

    strategies = {
        "PPO 动态策略": ppo_result,
        "智行均线策略": zhixing_result,
        "定投策略": dca_result,
    }

    if "Close" in df_prepared.columns:
        close_series = df_prepared["Close"].astype(float)
        if not close_series.empty:
            buy_and_hold = initial_cash * (close_series / close_series.iloc[0])
            buy_and_hold.name = "买入并持有"
            strategies["买入并持有"] = {
                "equity_curve": buy_and_hold,
                "metrics": {
                    "总回报率": float(buy_and_hold.iloc[-1] / initial_cash - 1.0),
                    "最终净值": float(buy_and_hold.iloc[-1]),
                },
                "trades_df": pd.DataFrame(),
                "actions_df": pd.DataFrame(),
                "raw": {},
            }

    composite_scores = _compute_composite_scores(strategies)
    for name, composite_score in composite_scores.items():
        strategies[name].setdefault("metrics", {})["综合评分"] = float(composite_score)

    summary = {name: data.get("metrics", {}) for name, data in strategies.items()}

    return {
        "strategies": strategies,
        "summary": summary,
        "meta": {
            "initial_cash": initial_cash,
            "monthly_investment": monthly_investment,
            "fee": fee,
        },
    }

if __name__ == '__main__':
    # 用于直接运行此脚本进行测试的示例
    try:
        data = pd.read_csv("data/sh510300.csv", parse_dates=['Date'])
        results = run_backtest(data)
        print("--- 智行均线策略回测结果 ---")
        print(results['metrics'])
    except FileNotFoundError:
        print("错误：请确保 'data/sh510300.csv' 文件存在以便进行测试。")
    except Exception as e:
        print(f"执行回测时发生错误: {e}")
