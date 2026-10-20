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

# Step 3 - train_epochs
import torch
import torch.nn as nn


def train_epochs(model, loaders, optimizer, epochs=2, scheduler=None, clip=None):
    criterion = nn.CrossEntropyLoss()
    device = next(model.parameters()).device

    history = {
        "train_loss": [],
        "val_acc": [],
    }

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    for epoch in range(epochs):
        # --- Training Loop ---
        model.train()
        running_train_loss = 0.0
        total_train_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            batch_size = inputs.size(0)
            running_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        mean_train_loss = running_train_loss / total_train_samples
        history["train_loss"].append(mean_train_loss)

        # --- Validation Evaluation ---
        model.eval()
        correct_val = 0
        total_val_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=-1)

                correct_val += (preds == targets).sum().item()
                total_val_samples += targets.size(0)

        val_accuracy = correct_val / total_val_samples
        history["val_acc"].append(val_accuracy)

    return history

# Step 4 - gradient_norms
def gradient_norms(model, xb, yb):
    # TODO: one backward pass; per-Linear weight-gradient norms from the input layer outward.
    model.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(xb), yb)
    loss.backward()
    return [float(l.weight.grad.norm()) for l in model.body if isinstance(l, nn.Linear)]

# Step 5 - apply_he_init
def apply_he_init(model):
    # TODO: kaiming_normal_(weight, nonlinearity='relu') and zero bias for every nn.Linear; return model.
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    return model

