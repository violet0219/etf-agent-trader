from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_policy
from src.data_loader import load_price_data
from src.features import build_latest_features
from src.data_quality import check_latest_features, has_error
from src.signal_engine import calculate_latest_scores
from src.portfolio_engine import build_target_weights


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading policy.yaml...")
    policy = load_policy("policy.yaml")

    signal_config = policy.get("signals", {})

    momentum_lookback = int(signal_config.get("momentum_lookback_months", 6))
    volatility_lookback = int(signal_config.get("volatility_lookback_months", 3))
    momentum_weight = float(signal_config.get("momentum_weight", 1.0))
    volatility_weight = float(signal_config.get("volatility_weight", 0.5))
    zscore_clip = float(signal_config.get("zscore_clip", 2.0))

    print("Loading price data...")
    prices = load_price_data(policy)

    print("Building latest features...")
    features = build_latest_features(
        prices=prices,
        momentum_lookback=momentum_lookback,
        volatility_lookback=volatility_lookback,
    )
    features.to_csv(output_dir / "latest_features.csv")

    print("Checking data quality...")
    issues = check_latest_features(features)
    issues.to_csv(output_dir / "data_quality_issues.csv", index=False)

    if issues.empty:
        print("No data quality issues found.")
    else:
        print()
        print("Data quality issues:")
        print(issues)

    if has_error(issues):
        raise RuntimeError("Data quality check failed with ERROR-level issues.")

    print("Calculating latest scores...")
    scores = calculate_latest_scores(
        features=features,
        momentum_weight=momentum_weight,
        volatility_weight=volatility_weight,
        zscore_clip=zscore_clip,
    )
    scores.to_csv(output_dir / "latest_scores.csv")

    print("Building target weights...")
    target_weights = build_target_weights(
        scores=scores,
        policy=policy,
    )
    target_weights.to_csv(output_dir / "target_weights.csv")

    print()
    print("Latest features:")
    print(features)

    print()
    print("Latest scores:")
    print(scores)

    print()
    print("Target weights:")
    print(target_weights)

    print()
    print("Target weight sum:", target_weights["target_weight"].sum())

    print()
    print("Saved files:")
    print("- outputs/latest_features.csv")
    print("- outputs/data_quality_issues.csv")
    print("- outputs/latest_scores.csv")
    print("- outputs/target_weights.csv")

    print()
    print("Decision test passed.")


if __name__ == "__main__":
    main()
