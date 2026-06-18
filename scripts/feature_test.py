from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_policy
from src.data_loader import load_price_data
from src.features import build_latest_features


def main() -> None:
    print("Loading policy.yaml...")
    policy = load_policy("policy.yaml")

    print("Loading price data...")
    prices = load_price_data(policy)

    print("Building latest features...")
    features = build_latest_features(
        prices=prices,
        momentum_lookback=6,
        volatility_lookback=3,
    )

    output_path = Path("outputs/latest_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path)

    print()
    print("Latest features:")
    print(features)

    print()
    print(f"Saved to {output_path}")
    print("Feature test passed.")


if __name__ == "__main__":
    main()
