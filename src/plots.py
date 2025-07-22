"""
Plotting script for accuracy results.
"""

import os
import json
import matplotlib.pyplot as plt

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "outputs", "experiment_results.json"
)
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


def plot_results():
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)
    plt.figure(figsize=(3, 3))
    for name, res in results.items():
        plt.scatter(res["acc_test"], res["acc_102"], label=name, s=40)
    plt.plot([10, 100], [10, 100], "k--", linewidth=3)
    plt.xlabel("CIFAR-10 test accuracy")
    plt.ylabel("CIFAR-10.2 accuracy")
    plt.title("CIFAR-10.2")
    plt.xlim(10, 100)
    plt.ylim(10, 100)
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, "cifar10_vs_cifar102.png")
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    plot_results()
