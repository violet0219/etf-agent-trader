from __future__ import annotations

import pandas as pd


def check_latest_features(
    features: pd.DataFrame,
    max_abs_momentum: float = 0.80,
    max_volatility: float = 0.30,
) -> pd.DataFrame:
    """
    Check latest feature table for suspicious values.

    This function does not decide whether to trade.
    It only reports warnings and errors.

    WARNING:
        Suspicious but not necessarily fatal.

    ERROR:
        Data problem serious enough to stop the pipeline.
    """
    required_columns = ["latest_price", "momentum", "volatility"]

    records = []

    for column in required_columns:
        if column not in features.columns:
            records.append(
                {
                    "level": "ERROR",
                    "asset_id": "",
                    "field": column,
                    "message": f"Missing required column: {column}",
                }
            )

    if records:
        return pd.DataFrame(records)

    for asset_id, row in features.iterrows():
        latest_price = row["latest_price"]
        momentum = row["momentum"]
        volatility = row["volatility"]

        if pd.isna(latest_price):
            records.append(
                {
                    "level": "ERROR",
                    "asset_id": asset_id,
                    "field": "latest_price",
                    "message": "Latest price is missing.",
                }
            )
        elif latest_price <= 0:
            records.append(
                {
                    "level": "ERROR",
                    "asset_id": asset_id,
                    "field": "latest_price",
                    "message": "Latest price must be positive.",
                }
            )

        if pd.isna(momentum):
            records.append(
                {
                    "level": "ERROR",
                    "asset_id": asset_id,
                    "field": "momentum",
                    "message": "Momentum is missing.",
                }
            )
        elif abs(momentum) > max_abs_momentum:
            records.append(
                {
                    "level": "WARNING",
                    "asset_id": asset_id,
                    "field": "momentum",
                    "message": (
                        f"Momentum looks unusually large: {momentum:.4f}. "
                        "Check whether this is a real move or a data issue."
                    ),
                }
            )

        if pd.isna(volatility):
            records.append(
                {
                    "level": "ERROR",
                    "asset_id": asset_id,
                    "field": "volatility",
                    "message": "Volatility is missing.",
                }
            )
        elif volatility > max_volatility:
            records.append(
                {
                    "level": "WARNING",
                    "asset_id": asset_id,
                    "field": "volatility",
                    "message": f"Volatility looks unusually high: {volatility:.4f}.",
                }
            )

    return pd.DataFrame(
        records,
        columns=["level", "asset_id", "field", "message"],
    )


def has_error(issues: pd.DataFrame) -> bool:
    """
    Return True if there is at least one ERROR-level issue.
    """
    if issues.empty:
        return False

    return bool((issues["level"] == "ERROR").any())
