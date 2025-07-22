"""
Experiment runner for CIFAR-10 and CIFAR-102 models.
"""

import sys
import os
import json
from typing import Dict, Any

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import load_cifar10, load_cifar102
from models.models import get_models, train_mlp

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "outputs")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_results(results: Dict[str, Any], path: str) -> None:
    """Save experiment results to a JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")
    except Exception as e:
        print(f"Error saving results: {e}")


def run_experiment() -> None:
    """Run experiments on CIFAR-10 and CIFAR-102 datasets with various models."""
    X_train, y_train, X_test, y_test = load_cifar10()
    X_102, y_102 = load_cifar102()
    results = {}
    # Sklearn models
    for name, model in get_models().items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        acc_test = model.score(X_test, y_test)
        acc_102 = model.score(X_102, y_102)
        results[name] = {"acc_test": acc_test * 100, "acc_102": acc_102 * 100}
    # MLP
    print("Training Neural Network...")
    acc_test, acc_102 = train_mlp(X_train, y_train, X_test, y_test, X_102, y_102)
    results["Neural Network"] = {"acc_test": acc_test * 100, "acc_102": acc_102 * 100}
    # Save results
    results_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    save_results(results, results_path)


if __name__ == "__main__":
    run_experiment()
