"""
Assignment 4: alpha-beta-CROWN verification on MNIST FC model
Tests robustness verification across multiple epsilon values.
"""
import subprocess
import sys
import os
import yaml

ABCROWN_PATH = "../alpha-beta-CROWN/complete_verifier/abcrown.py"
BASE_CONFIG = "mnist_fc_config.yaml"
EPSILONS = [0.001, 0.005, 0.01]

def run_verification(eps):
    """Run alpha-beta-CROWN for a given epsilon and return summary."""
    # Load base config and modify epsilon
    with open(BASE_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["specification"]["epsilon"] = eps
    cfg["bab"]["timeout"] = 100

    tmp_config = f"tmp_eps_{eps}.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f)

    print(f"\n{'='*50}")
    print(f"Running verification with epsilon = {eps}")
    print('='*50)

    result = subprocess.run(
        [sys.executable, ABCROWN_PATH, "--config", tmp_config],
        capture_output=True, text=True
    )

    os.remove(tmp_config)

    # Print summary lines
    for line in result.stdout.splitlines():
        if any(k in line for k in ["Final verified", "total verified", "total falsified",
                                    "mean time", "safe-incomplete", "unknown"]):
            print(line)

if __name__ == "__main__":
    print("alpha-beta-CROWN Verification Test")
    print(f"Model: mnist_fc.onnx")
    print(f"Dataset: MNIST (10 examples)")
    print(f"Epsilons: {EPSILONS}")

    for eps in EPSILONS:
        run_verification(eps)

    print("\nDone.")
