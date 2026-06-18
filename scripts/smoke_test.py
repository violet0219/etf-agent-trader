from pathlib import Path
import sys

# Make project root importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_policy
from src.data_loader import load_price_data, save_price_data


def main() -> None:
    print("Loading policy.yaml...")
    policy = load_policy("policy.yaml")

    print("Downloading price data...")
    prices = load_price_data(policy)

    print("Saving price data to outputs/prices.csv...")
    save_price_data(prices, "outputs/prices.csv")

    print()
    print("Price data shape:", prices.shape)
    print()
    print("Columns:")
    print(list(prices.columns))
    print()
    print("Last 5 rows:")
    print(prices.tail())

    if prices.empty:
        raise RuntimeError("Smoke test failed: price data is empty.")

    if prices.shape[1] == 0:
        raise RuntimeError("Smoke test failed: no price columns found.")

    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
