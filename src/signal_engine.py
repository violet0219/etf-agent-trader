from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_zscore(series: pd.Series, clip: float = 2.0) -> pd.Series:
    """
    Calculate cross-sectional z-score across assets.

    Example:
        Compare momentum values across all assets at the latest date.
    """
    values = series.astype(float).copy()

    mean = values.mean()
    std = values.std(ddof=0)

    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=values.index)

    zscore = (values - mean) / std
    zscore = zscore.clip(lower=-clip, upper=clip)

    return zscore


def calculate_latest_scores(
    features: pd.DataFrame,
    momentum_weight: float = 1.0,
    volatility_weight: float = 0.5,
    zscore_clip: float = 2.0,
) -> pd.DataFrame:
    """
    Calculate latest asset scores from feature table.

    Current v0.1 score:
        score = momentum_weight * momentum_z
                - volatility_weight * volatility_z

    High momentum is good.
    High volatility is penalized.
    """
    required_columns = ["momentum", "volatility"]

    for column in required_columns:
        if column not in features.columns:
            raise ValueError(f"Missing required feature column: {column}")

    momentum_z = cross_sectional_zscore(
        features["momentum"],
        clip=zscore_clip,
    )

    volatility_z = cross_sectional_zscore(
        features["volatility"],
        clip=zscore_clip,
    )

    score = momentum_weight * momentum_z - volatility_weight * volatility_z
    score = score.clip(lower=-zscore_clip, upper=zscore_clip)

    signals = pd.DataFrame(
        {
            "momentum": features["momentum"],
            "volatility": features["volatility"],
            "momentum_z": momentum_z,
            "volatility_z": volatility_z,
            "score": score,
        }
    )

    signals.index.name = "asset_id"

    return signals
