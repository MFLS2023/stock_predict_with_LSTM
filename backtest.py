import pandas as pd

try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
except ImportError as exc:
    raise ImportError(
        "依赖缺失：无法导入 'backtesting' 库，请先执行 `pip install backtesting` (或安装 requirements.txt) 后再运行 GUI。"
    ) from exc

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

def run_ai_comparison_backtest(*args, **kwargs):
    """AI对比回测的占位符函数。"""
    print("警告: run_ai_comparison_backtest() 尚未在此次恢复中实现。")
    raise NotImplementedError("AI对比回测功能待实现。")

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
