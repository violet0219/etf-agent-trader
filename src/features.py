from __future__ import annotations

import pandas as pd


def to_monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily prices to month-end prices.
    """
    if prices.empty:
        raise ValueError("Price data is empty.")

    monthly_prices = prices.resample("ME").last()
    monthly_prices = monthly_prices.dropna(how="all")

    return monthly_prices


def calculate_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly percentage returns from month-end prices.
    """
    returns = monthly_prices.pct_change()
    returns = returns.dropna(how="all")

    return returns


def calculate_momentum(monthly_prices: pd.DataFrame, lookback_months: int = 6) -> pd.DataFrame:
    """
    Calculate trailing momentum.

    Example:
        6-month momentum = current month-end price / price 6 months ago - 1
    """
    momentum = monthly_prices / monthly_prices.shift(lookback_months) - 1
    momentum = momentum.dropna(how="all")

    return momentum


def calculate_volatility(monthly_returns: pd.DataFrame, lookback_months: int = 3) -> pd.DataFrame:
    """
    Calculate trailing volatility using monthly returns.

    This is not annualized yet.
    """
    volatility = monthly_returns.rolling(lookback_months).std()
    volatility = volatility.dropna(how="all")

    return volatility


def build_latest_features(
    prices: pd.DataFrame,
    momentum_lookback: int = 6,
    volatility_lookback: int = 3,
) -> pd.DataFrame:
    """
    Build the latest feature table for each asset.

    Output index:
        asset id

    Output columns:
        latest_price
        momentum
        volatility
    """
    monthly_prices = to_monthly_prices(prices)
    monthly_returns = calculate_monthly_returns(monthly_prices)
    momentum = calculate_momentum(monthly_prices, momentum_lookback)
    volatility = calculate_volatility(monthly_returns, volatility_lookback)

    latest_price = monthly_prices.iloc[-1]
    latest_momentum = momentum.iloc[-1]
    latest_volatility = volatility.iloc[-1]

    features = pd.DataFrame(
        {
            "latest_price": latest_price,
            "momentum": latest_momentum,
            "volatility": latest_volatility,
        }
    )

    features.index.name = "asset_id"

    return features
