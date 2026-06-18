from pathlib import Path
import yaml


def load_policy(path: str = "policy.yaml") -> dict:
    """
    Load the portfolio policy YAML file.

    This file contains non-secret trading rules such as:
    - trading mode
    - broker name
    - risk limits
    - rebalance rules
    """
    policy_path = Path(path)

    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    with policy_path.open("r", encoding="utf-8") as file:
        policy = yaml.safe_load(file)

    if not isinstance(policy, dict):
        raise ValueError("Policy file must contain a YAML dictionary.")

    return policy


if __name__ == "__main__":
    policy = load_policy()

    print("Project:", policy.get("project", {}).get("name"))
    print("Trading mode:", policy.get("mode", {}).get("trading_mode"))
    print("Broker:", policy.get("broker", {}).get("name"))
