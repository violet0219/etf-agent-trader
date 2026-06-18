from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf


def get_price_ticker_map(policy: dict) -> Dict[str, str]:
    """
    Read asset IDs and price tickers from policy.yaml.

    Returns:
        {
            "us_equity": "SPY",
            "gold": "GLD",
            ...
        }
    """
    assets = policy.get("universe", {}).get("assets", [])

    ticker_map: Dict[str, str] = {}

    for asset in assets:
        asset_id = asset.get("id")
        ticker = asset.get("price_ticker")

        if asset_id and ticker:
            ticker_map[asset_id] = ticker

    if not ticker_map:
        raise ValueError("No price_ticker values found in policy.yaml.")

    return ticker_map


def download_adjusted_close(
    ticker_map: Dict[str, str],
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices from yfinance.

    Columns are renamed from tickers to asset IDs.
    Example:
        SPY -> us_equity
        GLD -> gold
    """
    tickers = list(ticker_map.values())

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        raise ValueError("Downloaded price data is empty.")

    # yfinance usually returns MultiIndex columns when multiple tickers are used.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Could not find Close prices in downloaded data.")

        prices = raw["Close"].copy()

    else:
        # Single ticker fallback
        if "Close" not in raw.columns:
            raise ValueError("Could not find Close prices in downloaded data.")

        prices = raw[["Close"]].copy()
        prices.columns = tickers

    ticker_to_asset_id = {ticker: asset_id for asset_id, ticker in ticker_map.items()}
    prices = prices.rename(columns=ticker_to_asset_id)

    prices = prices.sort_index()
    prices = prices.dropna(how="all")
    
    # Reorder columns according to policy.yaml order.
    asset_order = list(ticker_map.keys())
    prices = prices.reindex(columns=asset_order)

    # Remove leftover column name from yfinance output.
    prices.columns.name = None

    return prices


def load_price_data(policy: dict) -> pd.DataFrame:
    """
    Main entry point for loading price data.
    """
    data_config = policy.get("data", {})

    start_date = data_config.get("start_date")
    end_date = data_config.get("end_date")

    if not start_date:
        raise ValueError("data.start_date is required in policy.yaml.")

    ticker_map = get_price_ticker_map(policy)

    prices = download_adjusted_close(
        ticker_map=ticker_map,
        start_date=start_date,
        end_date=end_date,
    )

    return prices


def save_price_data(prices: pd.DataFrame, path: str = "outputs/prices.csv") -> None:
    """
    Save price data to CSV.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prices.to_csv(output_path, index=True)
