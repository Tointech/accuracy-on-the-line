"""
Experiment runner for CIFAR-10 and CIFAR-102 models.
"""

import sys
import os
import json
import random
import time
import argparse
import numpy as np
from typing import Any

import torch

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import load_cifar10, load_cifar102
from models.models import get_model, train_mlp

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "outputs")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PARAM_SPACES = {
    "Random Forest": lambda: {
        "n_estimators": np.random.choice([100, 200, 300]),
        "max_depth": np.random.choice([None, 10, 20, 30, 50]),
        "max_features": np.random.choice(["sqrt", "log2", None]),
        "n_jobs": -1,
    },
    "KNN": lambda: {
        "algorithm": "auto",
        "metric": "minkowski",
        "n_neighbors": np.random.choice([3, 5, 7, 9, 11]),
        "weights": np.random.choice(["uniform", "distance"]),
    },
    "Linear Model": lambda: {
        "solver": np.random.choice(["lbfgs", "saga", "newton-cg"]),
        "penalty": np.random.choice(["l2", None]),
        "C": np.random.uniform(0.01, 10.0),
        "max_iter": 1000,
    },
    "AdaBoost": lambda: {
        "n_estimators": np.random.choice([50, 100, 200]),
        "learning_rate": np.random.choice([0.01, 0.05, 0.1, 0.5, 1.0]),
    },
    "Random Features": lambda: {
        "gamma": np.random.uniform(0.1, 2.0),
        "n_components": np.random.choice([250, 500, 1000]),
        "lr_params": {
            "C": np.random.uniform(0.01, 10.0),
            "solver": "lbfgs",
            "max_iter": 1000,
        },
    },
}


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


def run_experiment(num_runs: int, results_path: str) -> None:
    """Run experiments on CIFAR-10 and CIFAR-102 datasets with various models."""
    X_train, y_train, X_test, y_test = load_cifar10()
    X_102, y_102 = load_cifar102()
    results = []

    for run in range(num_runs):
        print(f"\n=== Run {run + 1}/{num_runs} ===")
        seed = 47 + run
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Sklearn models
        for name in MODEL_PARAM_SPACES:
            params = MODEL_PARAM_SPACES[name]()
            for k, v in params.items():
                if isinstance(v, (np.integer, np.floating, np.str_)):
                    params[k] = v.item() if hasattr(v, "item") else v
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (np.integer, np.floating, np.str_)):
                            v[kk] = vv.item() if hasattr(vv, "item") else vv

            print(f"\nTraining {name} with params: {params}")
            start_time = time.time()
            model = get_model(name, params)
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
                    "params": {**params, "seed": seed},
                }
            )

        # Neural Network (MLP) with random hyperparameters
        hidden_dim = np.random.choice([128, 256, 512, 1024])
        lr = np.random.choice([1e-4, 1e-3, 1e-2, 0.05, 0.1])
        epochs = np.random.choice([10, 20, 30, 50])

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

    save_results(results, results_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CIFAR-10/CIFAR-102 model experiments."
    )
    parser.add_argument(
        "--num_runs", type=int, default=10, help="Number of runs per model."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="experiment_results.json",
        help="Results JSON path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    full_output_path = os.path.join(RESULTS_DIR, args.output_path)
    run_experiment(args.num_runs, full_output_path)
