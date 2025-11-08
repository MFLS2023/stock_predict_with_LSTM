from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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


@dataclass
class StrategyResult:
    """封装策略回测返回的统一结构。"""

    equity_curve: pd.Series
    metrics: Dict[str, Any]
    extra: Dict[str, Any]

def run_backtest(df: pd.DataFrame, initial_cash=100_000):
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

    # 运行回测
    bt = Backtest(bt_df, ZhixingMaStrategy, cash=initial_cash, commission=.001)
    stats = bt.run()
    
    # 返回一个包含关键结果的字典，方便GUI调用
    result = {
        "stats": stats,
        "trades": stats['_trades'],
        "equity_curve": stats['_equity_curve']['Equity'],
        "metrics": {
            "Total Return": stats['Return [%]'] / 100,
            "Annualized Return": stats['Return (Ann.) [%]'] / 100,
            "Max Drawdown": -stats['Max. Drawdown [%]'] / 100,
            "Sharpe Ratio": stats['Sharpe Ratio'],
            "Win Rate": stats['Win Rate [%]'] / 100,
            "Total Trades": stats['# Trades'],
            "Final Equity": stats['Equity Final [$]'],
        },
        # backtesting.py 的 plot() 函数需要原始的 _strategy 对象
        "_strategy": stats['_strategy'], 
        "_equity_curve_df": stats['_equity_curve']
    }
    # bt.plot() # 不在函数内直接绘图，由调用方决定
    return result

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


def run_ppo_backtest(
    model_path: str,
    df_test: pd.DataFrame,
    initial_cash: float,
    *,
    monthly_invest: float = 0.0,
    fee: float = 0.001,
) -> StrategyResult:
    """
    使用已训练好的 PPO 模型执行回测。

    返回值统一使用 StrategyResult 以便 GUI 层消费。
    """

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

    model = PPO.load(str(resolved_path))

    env = DynamicGridEnv(
        df=df_test.copy(),
        initial_cash=float(initial_cash),
        monthly_invest=float(monthly_invest),
        fee=float(fee),
    )

    obs, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    equity_curve = env.get_nav_series()
    if equity_curve.empty:
        equity_curve = pd.Series([initial_cash], name="AI策略")

    total_return = equity_curve.iloc[-1] / max(initial_cash, 1e-8) - 1.0

    metrics: Dict[str, Any] = {
        "Total Return": total_return,
        "Final Equity": equity_curve.iloc[-1],
        "Total Reward": env.total_reward,
        "Trades": len(env.trades),
    }

    extra: Dict[str, Any] = {
        "trades": env.trades,
        "cash_added": env.cash_added,
        "nav_history_raw": env.nav_history,
    }

    return StrategyResult(equity_curve=equity_curve, metrics=metrics, extra=extra)


def run_ai_comparison_backtest(
    model_path: str,
    df_test: pd.DataFrame,
    *,
    initial_cash: float = 100_000.0,
    fee: float = 0.001,
    monthly_invest: float = 0.0,
) -> Dict[str, Any]:
    """
    调度 PPO 策略与基准策略回测，并返回对比所需的全部结果。
    """

    print("开始执行策略对比回测...")

    df_prepared = _ensure_datetime_index(df_test)

    ppo_result = run_ppo_backtest(
        model_path=model_path,
        df_test=df_prepared,
        initial_cash=initial_cash,
        monthly_invest=monthly_invest,
        fee=fee,
    )

    baseline_df = df_prepared.reset_index().rename(columns={"index": "Date"}) if isinstance(df_prepared.index, pd.DatetimeIndex) else df_prepared.copy()
    zhixing_result_raw = run_backtest(baseline_df, initial_cash=initial_cash)
    zhixing_result = StrategyResult(
        equity_curve=zhixing_result_raw["equity_curve"],
        metrics=zhixing_result_raw["metrics"],
        extra={"stats": zhixing_result_raw.get("stats"), "trades": zhixing_result_raw.get("trades")},
    )

    if "Close" not in df_prepared.columns:
        raise ValueError("测试数据缺少 `Close` 列，无法计算买入并持有曲线。")

    close_series = df_prepared["Close"].astype(float)
    buy_and_hold = initial_cash * (close_series / close_series.iloc[0])
    buy_and_hold.name = "买入并持有"

    metrics_buy_hold = {
        "Total Return": buy_and_hold.iloc[-1] / initial_cash - 1.0,
        "Final Equity": buy_and_hold.iloc[-1],
    }

    equity_curves = {
        "PPO 动态策略": ppo_result.equity_curve,
        "智行均线策略": zhixing_result.equity_curve,
        "买入并持有": buy_and_hold,
    }

    metrics_combined = {
        "PPO 动态策略": ppo_result.metrics,
        "智行均线策略": zhixing_result.metrics,
        "买入并持有": metrics_buy_hold,
    }

    return {
        "equity_curves": equity_curves,
        "metrics": metrics_combined,
        "details": {
            "ppo": ppo_result.extra,
            "baseline": zhixing_result.extra,
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
