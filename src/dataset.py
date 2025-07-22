"""
Dataset loading and preprocessing
"""

from torchvision import datasets, transforms
import numpy as np
import os


# 1. Data Loading and Preprocessing
def load_cifar10():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = datasets.CIFAR10(
        root="./data/raw", train=True, download=True, transform=transform
    )
    testset = datasets.CIFAR10(
        root="./data/raw", train=False, download=True, transform=transform
    )
    X_train = trainset.data.reshape(len(trainset), -1) / 255.0
    y_train = np.array(trainset.targets)
    X_test = testset.data.reshape(len(testset), -1) / 255.0
    y_test = np.array(testset.targets)
    return X_train, y_train, X_test, y_test


def load_cifar102():
    # Download from https://github.com/modestyachts/CIFAR-10.2
    # Assume user has placed cifar102_test.npz in ./data/
    cifar102_path = "./data/raw/cifar102_test.npz"
    if not os.path.exists(cifar102_path):
        raise FileNotFoundError(
            "Download CIFAR-10.2 test set from https://github.com/modestyachts/cifar-10.2 and place cifar102_test.npz in ./data/"
        )
    data = np.load(cifar102_path)
    X = data["images"].reshape(len(data["images"]), -1) / 255.0
    y = data["labels"]
    return X, y
