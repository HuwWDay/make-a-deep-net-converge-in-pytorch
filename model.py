"""
Make a Deep Net Converge in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - fashion_loaders
import gzip
import os
import tempfile
import urllib.request
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def fashion_loaders(n_train=5000, n_val=1000, batch_size=64, seed=42):
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    temp_dir = tempfile.gettempdir()
    local_paths = {}

    # Download and cache files
    for key, filename in files.items():
        dest_path = os.path.join(temp_dir, f"fashion_{filename}")
        if not os.path.exists(dest_path):
            url = f"{base_url}{filename}"
            urllib.request.urlretrieve(url, dest_path)
        local_paths[key] = dest_path

    # Parse raw idx-ubyte buffers
    with gzip.open(local_paths["train_images"], "rb") as f:
        # Offset 16 bytes: magic number (4), count (4), rows (4), cols (4)
        raw_images = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        raw_images = raw_images.reshape(-1, 28 * 28)

    with gzip.open(local_paths["train_labels"], "rb") as f:
        # Offset 8 bytes: magic number (4), count (4)
        raw_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    # Scale to [0, 1] and standardize
    images = raw_images.astype(np.float32) / 255.0
    mean = 0.2860
    std = 0.3530
    images = (images - mean) / std

    # Split train and validation slices
    train_images = images[:n_train]
    train_labels = raw_labels[:n_train].astype(np.int64)

    val_images = images[n_train : n_train + n_val]
    val_labels = raw_labels[n_train : n_train + n_val].astype(np.int64)

    # Convert to PyTorch TensorDatasets
    train_dataset = TensorDataset(
        torch.from_numpy(train_images), torch.from_numpy(train_labels)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_images), torch.from_numpy(val_labels)
    )

    # Seeded generator for reproducible train shuffle
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "n_features": 784,
    }

# Step 2 - DeepNet
import torch.nn as nn


class DeepNet(nn.Module):
    def __init__(
        self,
        n_features=784,
        n_hidden=100,
        n_layers=10,
        activation="sigmoid",
        batchnorm=False,
        dropout=0.0,
        n_classes=10,
    ):
        super().__init__()

        # Select activation module
        act_lower = activation.lower()
        if act_lower == "sigmoid":
            act_cls = nn.Sigmoid
        elif act_lower == "relu":
            act_cls = nn.ReLU
        else:
            raise ValueError(f"Unsupported activation: '{activation}'. Expected 'sigmoid' or 'relu'.")

        layers = []
        in_dim = n_features

        for _ in range(n_layers):
            # 1. Linear projection
            layers.append(nn.Linear(in_dim, n_hidden))

            # 2. Optional BatchNorm1d
            if batchnorm:
                layers.append(nn.BatchNorm1d(n_hidden))

            # 3. Activation
            layers.append(act_cls())

            # 4. Optional Dropout
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

            # Subsequent hidden layers map from n_hidden to n_hidden
            in_dim = n_hidden

        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        features = self.body(x)
        logits = self.head(features)
        return logits

