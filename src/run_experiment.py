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
from models.models import get_model, train_mlp

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "outputs")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_RUNS = 1  # Number of repeated runs per model
N_ESTIMATORS = [50, 100, 150, 200]
LEARNING_RATES = [0.001, 0.01, 0.1, 1, 10]

MODEL_PARAM_SPACES = {
    # "Random Forest": lambda: {
    #     "n_estimators": np.random.choice(N_ESTIMATORS),
    #     "max_depth": np.random.choice([None, 10, 20, 30]),
    #     "n_jobs": -1,
    # },
    "KNN": lambda: {
        "n_neighbors": np.random.choice([3, 5, 7]),
        "weights": np.random.choice(["uniform", "distance"]),
        "algorithm": np.random.choice(["auto", "ball_tree", "kd_tree", "brute"]),
        "metric": np.random.choice(["minkowski", "euclidean", "manhattan"]),
    },
    "Linear Model": lambda: (
        lambda solver_penalty: {
            "max_iter": 1000,
            "solver": solver_penalty[0],
            "penalty": solver_penalty[1],
        }
    )(
        random.choice(
            [
                ("lbfgs", "l2"),
                ("lbfgs", None),
                ("liblinear", "l1"),
                ("liblinear", "l2"),
                ("newton-cg", "l2"),
                ("newton-cg", None),
                ("newton-cholesky", "l2"),
                ("newton-cholesky", None),
                ("saga", "l1"),
                ("saga", "l2"),
                ("saga", "elasticnet"),
                ("saga", None),
            ]
        )
    ),
    "AdaBoost": lambda: {
        "n_estimators": np.random.choice(N_ESTIMATORS),
        "learning_rate": np.random.choice(LEARNING_RATES),
    },
    "Random Features": lambda: {
        "gamma": np.random.uniform(0.5, 2.0),
        "n_components": np.random.choice([250, 500, 1000]),
        "lr_params": {"max_iter": 1000},
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


def run_experiment() -> None:
    """Run experiments on CIFAR-10 and CIFAR-102 datasets with various models."""
    X_train, y_train, X_test, y_test = load_cifar10()
    X_102, y_102 = load_cifar102()
    results = []

    for run in range(N_RUNS):
        print(f"\n=== Run {run + 1}/{N_RUNS} ===")
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
            # Convert all np types to native Python types
            for k, v in params.items():
                if isinstance(v, (np.integer, np.int32, np.int64)):
                    params[k] = int(v)
                elif isinstance(v, (np.floating, np.float32, np.float64)):
                    params[k] = float(v)
                elif isinstance(v, (np.str_,)):
                    params[k] = str(v)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (np.integer, np.int32, np.int64)):
                            v[kk] = int(vv)
                        elif isinstance(vv, (np.floating, np.float32, np.float64)):
                            v[kk] = float(vv)
                        elif isinstance(vv, (np.str_,)):
                            v[kk] = str(vv)
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
        hidden_dim = np.random.choice([256, 512, 1024])
        lr = np.random.choice(LEARNING_RATES)
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
    results_path = os.path.join(RESULTS_DIR, "experiment_results_refactor_1.json")
    save_results(results, results_path)


if __name__ == "__main__":
    run_experiment()
