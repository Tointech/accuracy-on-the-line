"""
Plotting script for accuracy results between CIFAR-10 and CIFAR-10.2.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot CIFAR-10 vs CIFAR-10.2 accuracy."
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "models",
            "outputs",
            "experiment_results_refactor.json",
        ),
        help="Path to experiment results JSON file.",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="cifar10_vs_cifar102_refactor.png",
        help="Filename for the saved plot image.",
    )
    return parser.parse_args()


def plot_results(results_path: str, save_name: str):
    with open(results_path, "r") as f:
        results = json.load(f)

    X = np.array([r["acc_test"] for r in results]).reshape(-1, 1)
    y = np.array([r["acc_102"] for r in results])

    # Linear regression fit
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = r2_score(y, reg.predict(X))

    # Plot
    plt.figure(figsize=(5, 5))
    model_labels = set()
    for r in results:
        label = r["model"] if r["model"] not in model_labels else None
        plt.scatter(r["acc_test"], r["acc_102"], label=label, s=30, alpha=0.6)
        model_labels.add(r["model"])

    x_vals = np.linspace(10, 100, 100)
    plt.plot(x_vals, x_vals, "k--", label="y = x")
    plt.plot(
        x_vals,
        reg.predict(x_vals.reshape(-1, 1)),
        "r-",
        label=f"Linear Fit\nSlope: {slope:.2f}, $R^2$: {r2:.2f}",
    )

    plt.xlabel("CIFAR-10 test accuracy")
    plt.ylabel("CIFAR-10.2 accuracy")
    plt.title("Accuracy on the Line")
    plt.xlim(10, 100)
    plt.ylim(10, 100)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    fig_path = os.path.join(FIGURE_DIR, save_name)
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    args = parse_args()
    plot_results(args.results_path, args.save_name)
