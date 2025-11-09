import unittest

import pandas as pd

from ai_strategy import DynamicGridEnv


def _sample_dataframe() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    base_price = 10.0
    close = [base_price + (i % 5 - 2) * 0.3 for i in range(len(dates))]
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": [c * 1.01 for c in close],
            "Low": [c * 0.99 for c in close],
            "Close": close,
            "Volume": [1_000] * len(close),
        }
    )
    frame["Amount"] = frame["Close"] * frame["Volume"]
    return frame


class RiskManagementTests(unittest.TestCase):
    def test_stop_loss_not_lowered_after_additional_buy(self) -> None:
        env = DynamicGridEnv(_sample_dataframe(), initial_cash=100_000.0, fee=0.0)
        env.reset()

        env.holding_shares = 100.0
        env._position_cost = 1_000.0
        env.stop_loss_price = 9.0
        env.take_profit_price = 11.0
        env.average_entry_price = 10.0

        current_price = 9.5
        shares_to_buy = 10.0
        trade_amount = current_price * shares_to_buy
        commission = env._calculate_commission(trade_amount)

        env.holding_shares += shares_to_buy
        env._position_cost += trade_amount + commission

        env._update_risk_targets_after_buy(
            current_price=current_price,
            stop_loss_pct=0.1,
            take_profit_pct=0.2,
        )

        self.assertIsNotNone(env.stop_loss_price)
        self.assertGreaterEqual(env.stop_loss_price or 0.0, 9.0 - 1e-6)
        self.assertLess(env.average_entry_price, 10.0)

    def test_partial_sell_trails_to_breakeven(self) -> None:
        env = DynamicGridEnv(_sample_dataframe(), initial_cash=100_000.0, fee=0.0)
        env.reset()

        env.holding_shares = 120.0
        env._position_cost = 1_200.0
        env.stop_loss_price = 9.0
        env.average_entry_price = 10.0

        current_price = 11.0
        shares_to_sell = 40.0
        proceeds = shares_to_sell * current_price
        commission = env._calculate_commission(proceeds)
        ratio = shares_to_sell / 120.0
        cost_released = env._position_cost * ratio
        net_proceeds = proceeds - commission
        profit = net_proceeds - cost_released

        env.holding_shares -= shares_to_sell
        env._position_cost = max(env._position_cost - cost_released, 0.0)
        env.cash += net_proceeds
        env._refresh_average_entry_price()

        if profit > 0:
            env._maybe_raise_stop_to_breakeven(current_price)

        self.assertGreater(env.holding_shares, 0.0)
        self.assertIsNotNone(env.stop_loss_price)
        self.assertGreaterEqual((env.stop_loss_price or 0.0) + 1e-6, env.average_entry_price)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
