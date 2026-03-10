import sys
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


# 1. Metadata
# ---------------------------------------------------------------------------

def get_task_metadata() -> dict:
    return {
        "task_id":            "logreg_lvl2_multiclass_softmax",
        "series":             "Logistic Regression",
        "level":              2,
        "algorithm":          "Softmax Regression (Multiclass)",
        "interface_protocol": "pytorch_task_v1",
    }

# 2. Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 3. Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. Data
# ---------------------------------------------------------------------------

def make_dataloaders(
    n_samples: int = 900,
    n_features: int = 2,
    n_classes: int = 3,
    cluster_std: float = 0.9,
    val_frac: float = 0.2,
    batch_size: int = 64,
    seed: int = 42,
):
    set_seed(seed)
    device = get_device()

    X_np, y_np = make_blobs(
        n_samples   = n_samples,
        n_features  = n_features,
        centers     = n_classes,
        cluster_std = cluster_std,
        random_state= seed,
    )

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    perm = torch.randperm(X.size(0))
    X, y = X[perm], y[perm]

    n_val   = int(n_samples * val_frac)
    n_train = n_samples - n_val

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    mean = X_train.mean(dim=0)
    std  = X_train.std(dim=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std

    X_val_np = X_val.numpy().copy()
    y_val_np = y_val.numpy().copy()

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val   = X_val.to(device)
    y_val   = y_val.to(device)

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds   = torch.utils.data.TensorDataset(X_val,   y_val)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_val_np, y_val_np

# 5. Model — linear softmax (single nn.Linear, no hidden layers)
# ---------------------------------------------------------------------------

class SoftmaxRegression(nn.Module):

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)   # [N, n_classes] logits


def build_model(
    n_features: int = 2,
    n_classes: int = 3,
    device: torch.device = None,
) -> nn.Module:
    if device is None:
        device = get_device()
    model = SoftmaxRegression(n_features, n_classes).to(device)
    return model

# 6. Train
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history     = []
    val_loss_history = []

    for _epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        total_n    = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)
            total_n    += X_batch.size(0)

        train_loss = epoch_loss / total_n

        model.eval()
        val_loss = 0.0
        val_n    = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits    = model(X_batch)
                val_loss += criterion(logits, y_batch).item() * X_batch.size(0)
                val_n    += X_batch.size(0)
        val_loss /= val_n

        loss_history.append(train_loss)
        val_loss_history.append(val_loss)

    return {
        "loss_history":     loss_history,
        "val_loss_history": val_loss_history,
    }

# 7. Evaluate
# ---------------------------------------------------------------------------

def _macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int) -> float:
    f1_scores = []
    for k in range(n_classes):
        tp = ((y_pred == k) & (y_true == k)).sum().item()
        fp = ((y_pred == k) & (y_true != k)).sum().item()
        fn = ((y_pred != k) & (y_true == k)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    n_classes: int = 3,
    split_name: str = "val",
) -> dict:
    criterion = nn.CrossEntropyLoss()
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            all_logits.append(model(X_batch))
            all_labels.append(y_batch)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    loss     = criterion(logits, labels).item()
    preds    = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean().item()
    macro_f1 = _macro_f1(labels.cpu(), preds.cpu(), n_classes)

    return {
        "split":     split_name,
        "loss":      loss,
        "accuracy":  accuracy,
        "macro_f1":  macro_f1,
    }

# 8. Predict
# ---------------------------------------------------------------------------

def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(X.to(device))
    return logits.argmax(dim=1)

# 9. Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    model: nn.Module,
    X_val_np: np.ndarray,
    y_val_np: np.ndarray,
    history: dict,
    n_classes: int = 3,
    output_dir: str = ".",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

    x_min, x_max = X_val_np[:, 0].min() - 1.0, X_val_np[:, 0].max() + 1.0
    y_min, y_max = X_val_np[:, 1].min() - 1.0, X_val_np[:, 1].max() + 1.0
    h = 0.05
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )
    grid = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        Z = model(grid).argmax(dim=1).cpu().numpy()
    Z = Z.reshape(xx.shape)

    colors_bg   = ["#AEC6CF", "#FFD1BA", "#C5E8C5"]
    colors_pts  = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=colors_bg[:n_classes], alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0.5, 1.5], colors="k", linewidths=0.8,
               linestyles="--")

    for k in range(n_classes):
        mask = y_val_np == k
        ax.scatter(
            X_val_np[mask, 0], X_val_np[mask, 1],
            c=colors_pts[k], label=f"Class {k}",
            edgecolors="k", s=30, linewidths=0.4, alpha=0.85,
        )

    ax.set_xlabel("Feature 1 (standardised)")
    ax.set_ylabel("Feature 2 (standardised)")
    ax.set_title("Softmax Regression — Decision Boundary (Validation Data)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = os.path.join(output_dir, "logreg_lvl2_boundary.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")

# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    CFG = {
        "n_samples":    900,
        "n_features":   2,
        "n_classes":    3,
        "cluster_std":  0.9,
        "val_frac":     0.2,
        "batch_size":   64,
        "n_epochs":     200,
        "lr":           1e-2,
        "weight_decay": 1e-4,
        "output_dir":   ".",
        "min_macro_f1": 0.85,
        "max_loss":     0.50,
    }

    print("=" * 60)
    print(f"Task  : {get_task_metadata()['task_id']}")
    print(f"Device: {device}")
    print("=" * 60)

    # Data
    train_loader, val_loader, X_val_np, y_val_np = make_dataloaders(
        n_samples   = CFG["n_samples"],
        n_features  = CFG["n_features"],
        n_classes   = CFG["n_classes"],
        cluster_std = CFG["cluster_std"],
        val_frac    = CFG["val_frac"],
        batch_size  = CFG["batch_size"],
        seed        = 42,
    )

    # Model
    model = build_model(
        n_features = CFG["n_features"],
        n_classes  = CFG["n_classes"],
        device     = device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel : SoftmaxRegression  ({n_params} parameters)")

    # Train
    print(f"\nTraining for {CFG['n_epochs']} epochs  "
          f"(Adam, lr={CFG['lr']}, wd={CFG['weight_decay']}) ...")
    history = train(
        model,
        train_loader,
        val_loader,
        n_epochs     = CFG["n_epochs"],
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
    )

    # Evaluate
    train_metrics = evaluate(
        model, train_loader, n_classes=CFG["n_classes"], split_name="train")
    val_metrics   = evaluate(
        model, val_loader,   n_classes=CFG["n_classes"], split_name="val")

    print("\n--- Train Metrics ---")
    print(f"  Loss     : {train_metrics['loss']:.6f}")
    print(f"  Accuracy : {train_metrics['accuracy']:.4f}")
    print(f"  Macro-F1 : {train_metrics['macro_f1']:.4f}")

    print("\n--- Validation Metrics ---")
    print(f"  Loss     : {val_metrics['loss']:.6f}")
    print(f"  Accuracy : {val_metrics['accuracy']:.4f}")
    print(f"  Macro-F1 : {val_metrics['macro_f1']:.4f}")

    # Learned parameters
    W = model.linear.weight.data.cpu()
    b = model.linear.bias.data.cpu()
    print(f"\n--- Learned Parameters ---")
    print(f"  W (shape {list(W.shape)}):\n{W.numpy().round(4)}")
    print(f"  b : {b.numpy().round(4)}")

    # Save artifacts
    print("\n--- Saving Artifacts ---")
    save_artifacts(
        model, X_val_np, y_val_np, history,
        n_classes  = CFG["n_classes"],
        output_dir = CFG["output_dir"],
    )

    # Loss convergence check
    val_losses  = history["val_loss_history"]
    warmup_end  = max(1, int(len(val_losses) * 0.10))
    post_warmup = val_losses[warmup_end:]

    def _moving_avg(seq, window=15):
        out = []
        for i in range(len(seq)):
            lo = max(0, i - window // 2)
            hi = min(len(seq), i + window // 2 + 1)
            out.append(sum(seq[lo:hi]) / (hi - lo))
        return out

    smoothed   = _moving_avg(post_warmup)
    seg_len    = max(1, len(smoothed) // 5)
    early_avg  = sum(smoothed[:seg_len]) / seg_len
    late_avg   = sum(smoothed[-seg_len:]) / seg_len
    rel_change = (late_avg - early_avg) / (early_avg + 1e-12)
    mono_ok    = rel_change <= 0.05

    print(f"\n--- Loss Convergence Check ---")
    print(f"  Early-seg avg : {early_avg:.6f}  (first {seg_len} smoothed epochs)")
    print(f"  Late-seg  avg : {late_avg:.6f}  (last  {seg_len} smoothed epochs)")
    print(f"  Rel change    : {rel_change:+.4f}  (tolerance: +0.05)")
    print(f"  Result        : {'PASS' if mono_ok else 'FAIL'}")

    # Quality assertions
    print("\n--- Quality Assertions ---")
    try:
        assert val_metrics["macro_f1"] > CFG["min_macro_f1"], (
            f"Val Macro-F1 {val_metrics['macro_f1']:.4f} <= threshold {CFG['min_macro_f1']}"
        )
        print(f"  [PASS] Val Macro-F1 {val_metrics['macro_f1']:.4f} > {CFG['min_macro_f1']}")

        assert val_metrics["loss"] < CFG["max_loss"], (
            f"Val loss {val_metrics['loss']:.4f} >= threshold {CFG['max_loss']}"
        )
        print(f"  [PASS] Val loss {val_metrics['loss']:.4f} < {CFG['max_loss']}")

        assert mono_ok, (
            f"Loss did not converge: late-seg avg {late_avg:.6f} > "
            f"early-seg avg {early_avg:.6f} by {rel_change:+.4f}"
        )
        print(f"  [PASS] Loss converges (segment rel change within tolerance)")

    except AssertionError as err:
        print(f"\n  [FAIL] {err}")
        sys.exit(1)

    print("\n[SUCCESS] All checks passed. Exiting 0.")
    sys.exit(0)
