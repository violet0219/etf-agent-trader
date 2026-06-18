from __future__ import annotations

import numpy as np
import pandas as pd


def softmax(scores: pd.Series, temperature: float = 1.0) -> pd.Series:
    """
    Convert scores into positive raw weights.

    Higher score -> higher weight.
    Higher temperature -> smoother, less concentrated allocation.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    values = scores.astype(float).fillna(0.0)
    scaled = values / temperature

    # Numerical stability
    scaled = scaled - scaled.max()

    exp_values = np.exp(np.clip(scaled, -50, 50))
    weights = exp_values / exp_values.sum()

    return pd.Series(weights, index=scores.index)


def get_asset_caps(policy: dict, asset_ids: list[str]) -> pd.Series:
    """
    Read per-asset max weights from policy.yaml.

    Final cap for each asset:
        min(asset.max_weight, risk.max_single_asset_weight)
    """
    assets = policy.get("universe", {}).get("assets", [])
    risk_config = policy.get("risk", {})

    global_max = float(risk_config.get("max_single_asset_weight", 1.0))

    asset_max_map = {}

    for asset in assets:
        asset_id = asset.get("id")
        max_weight = asset.get("max_weight", 1.0)

        if asset_id:
            asset_max_map[asset_id] = min(float(max_weight), global_max)

    caps = pd.Series(index=asset_ids, dtype=float)

    for asset_id in asset_ids:
        caps.loc[asset_id] = asset_max_map.get(asset_id, global_max)

    return caps


def apply_max_weight_caps(
    raw_weights: pd.Series,
    caps: pd.Series,
    total_weight: float,
    tolerance: float = 1e-10,
) -> pd.Series:
    """
    Apply max weight constraints and redistribute excess weight.

    Example:
        raw says korea_equity should be 60%,
        but cap is 25%.
        Then korea_equity is capped at 25%,
        and the remaining weight is redistributed to other assets.
    """
    if total_weight < 0 or total_weight > 1:
        raise ValueError("total_weight must be between 0 and 1.")

    raw_weights = raw_weights.astype(float).copy()
    caps = caps.astype(float).reindex(raw_weights.index)

    if caps.isna().any():
        missing = list(caps[caps.isna()].index)
        raise ValueError(f"Missing caps for assets: {missing}")

    if caps.sum() + tolerance < total_weight:
        raise ValueError(
            "Sum of max weight caps is smaller than target total weight. "
            "Relax max_weight constraints or reduce min_cash_weight."
        )

    if raw_weights.sum() <= 0:
        raw_weights = pd.Series(1.0 / len(raw_weights), index=raw_weights.index)
    else:
        raw_weights = raw_weights / raw_weights.sum()

    final_weights = pd.Series(0.0, index=raw_weights.index)
    active_assets = list(raw_weights.index)
    remaining_weight = total_weight

    while active_assets:
        active_raw = raw_weights.loc[active_assets]

        if active_raw.sum() <= 0:
            tentative = pd.Series(
                remaining_weight / len(active_assets),
                index=active_assets,
            )
        else:
            tentative = active_raw / active_raw.sum() * remaining_weight

        active_caps = caps.loc[active_assets]
        breached = tentative > active_caps + tolerance

        if not breached.any():
            final_weights.loc[active_assets] = tentative
            break

        capped_assets = list(breached[breached].index)

        for asset_id in capped_assets:
            final_weights.loc[asset_id] = active_caps.loc[asset_id]
            remaining_weight -= active_caps.loc[asset_id]

        active_assets = [
            asset_id for asset_id in active_assets if asset_id not in capped_assets
        ]

    return final_weights


def build_target_weights(scores: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """
    Convert latest scores into target portfolio weights.

    Output includes a separate cash row if risk.min_cash_weight > 0.
    """
    if "score" not in scores.columns:
        raise ValueError("scores DataFrame must contain a 'score' column.")

    risk_config = policy.get("risk", {})
    allocation_config = policy.get("allocation", {})

    min_cash_weight = float(risk_config.get("min_cash_weight", 0.0))
    investable_weight = 1.0 - min_cash_weight

    if investable_weight <= 0:
        raise ValueError("Investable weight must be positive.")

    temperature = float(allocation_config.get("softmax_temperature", 1.0))

    raw_asset_weights = softmax(
        scores["score"],
        temperature=temperature,
    )

    asset_ids = list(scores.index)
    caps = get_asset_caps(policy, asset_ids)

    target_asset_weights = apply_max_weight_caps(
        raw_weights=raw_asset_weights,
        caps=caps,
        total_weight=investable_weight,
    )

    target = pd.DataFrame(
        {
            "raw_weight": raw_asset_weights,
            "max_weight": caps,
            "target_weight": target_asset_weights,
        }
    )

    target.index.name = "asset_id"

    if min_cash_weight > 0:
        target.loc["cash"] = {
            "raw_weight": 0.0,
            "max_weight": 1.0,
            "target_weight": min_cash_weight,
        }

    target["target_weight_pct"] = target["target_weight"] * 100

    return target
