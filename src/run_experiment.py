"""
Experiment runner for CIFAR-10 and CIFAR-102 models.
"""

import sys
import os
import json
import random
import time
import numpy as np
from typing import Any

import torch

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import load_cifar10, load_cifar102
from models.models import get_models, train_mlp

DAY = "22_07"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "outputs")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_RUNS = 5  # Number of repeated runs per model


def save_results(results: Any, path: str) -> None:
    """Save experiment results to a JSON file (with NumPy types handled)."""

    def convert(o):
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        elif isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        elif isinstance(o, (np.ndarray,)):
            return o.tolist()
        return o

    try:
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=convert)
        print(f"Results saved to {path}")
    except Exception as e:
        print(f"Error saving results: {e}")


def run_experiment() -> None:
    """Run experiments on CIFAR-10 and CIFAR-102 datasets with various models."""
    X_train, y_train, X_test, y_test = load_cifar10()
    X_102, y_102 = load_cifar102()
    results = []

    for run in range(N_RUNS):
        print(f"\n=== Run {run + 1}/{N_RUNS} ===")
        seed = 42 + run
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Sklearn models
        for name, model in get_models().items():
            print(f"\nTraining {name}...")
            start_time = time.time()

            model.fit(X_train, y_train)
            acc_test = model.score(X_test, y_test)
            acc_102 = model.score(X_102, y_102)

            duration = time.time() - start_time
            print(
                f"{name} - acc_test: {acc_test * 100:.2f}%, acc_102: {acc_102 * 100:.2f}%, time: {duration:.2f}s"
            )

            results.append(
                {
                    "model": name,
                    "run": run,
                    "acc_test": acc_test * 100,
                    "acc_102": acc_102 * 100,
                    "time": duration,
                    "params": {"seed": seed},
                }
            )

        # Neural Network (MLP) with random hyperparameters
        hidden_dim = np.random.choice([256, 512, 1024])
        lr = 10 ** np.random.uniform(-4, -2)  # log-uniform from 1e-4 to 1e-2
        epochs = np.random.choice([5, 10, 15])

        print(
            f"\nTraining Neural Network (hidden_dim={hidden_dim}, lr={lr:.4f}, epochs={epochs})"
        )
        start_time = time.time()

        acc_test, acc_102 = train_mlp(
            X_train,
            y_train,
            X_test,
            y_test,
            X_102,
            y_102,
            hidden_dim=hidden_dim,
            lr=lr,
            epochs=epochs,
        )

        duration = time.time() - start_time
        print(
            f"Neural Network - acc_test: {acc_test * 100:.2f}%, acc_102: {acc_102 * 100:.2f}%, time: {duration:.2f}s"
        )

        results.append(
            {
                "model": "Neural Network",
                "run": run,
                "acc_test": acc_test * 100,
                "acc_102": acc_102 * 100,
                "time": duration,
                "params": {
                    "hidden_dim": hidden_dim,
                    "lr": lr,
                    "epochs": epochs,
                    "seed": seed,
                },
            }
        )

    # Save results
    results_path = os.path.join(RESULTS_DIR, f"experiment_results_{DAY}.json")
    save_results(results, results_path)


if __name__ == "__main__":
    run_experiment()
