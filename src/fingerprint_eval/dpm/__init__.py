"""Dynamic-parimutuel market (DPM) core.

Faithful implementation of the Delphi L2 / Euclidean-norm dynamic parimutuel
market, transcribed from the verified ``DynamicParimutuelMath`` contract source.
See ``dpm_spec.md`` for the exact math this mirrors.
"""

from ..market import (
    Market,
    MarketState,
    Order,
    OutcomeInfo,
    TradeResult,
)
from .market import CREATOR_ADDRESS, DEFAULT_TRADER, MarketConfig, PriceBucketMarket
from .math import (
    buy_cost,
    collateral,
    implied_probabilities,
    liquidation_reward,
    sell_proceeds,
    settlement_payout,
    shares_for_buy_spend,
    shares_for_sell_proceeds,
    spot_prices,
    sum_term,
)

__all__ = [
    # math
    "sum_term",
    "collateral",
    "spot_prices",
    "implied_probabilities",
    "buy_cost",
    "sell_proceeds",
    "shares_for_buy_spend",
    "shares_for_sell_proceeds",
    "settlement_payout",
    "liquidation_reward",
    # market
    "MarketConfig",
    "PriceBucketMarket",
    "CREATOR_ADDRESS",
    "DEFAULT_TRADER",
    # seam (re-exported for convenience)
    "Market",
    "MarketState",
    "Order",
    "OutcomeInfo",
    "TradeResult",
]
