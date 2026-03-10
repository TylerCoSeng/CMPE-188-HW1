"""
Task: logreg_lvl1_binary_raw
Series: Logistic Regression | Level 1
Algorithm: Logistic Regression (Binary, Raw Tensors)

Math
----
Sigmoid (logistic) function:
    sigma(z) = 1 / (1 + e^{-z})

Prediction:
    z        = X @ w + b
    y_hat    = sigma(z)

Binary cross-entropy (log-loss) over N samples:
    J(w, b) = -(1/N) * sum[ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]

Manual gradients (derived from dJ/dw, dJ/db):
    dJ/dw = (1/N) * X^T @ (y_hat - y)
    dJ/db = (1/N) * sum(y_hat - y)

Weight update (pure gradient descent, NO autograd, NO torch.optim):
    w = w - lr * dJ/dw
    b = b - lr * dJ/db

Saves
-----
    logreg_lvl1_loss.png  — train vs validation log-loss curve
"""

import sys
import os
import random
import math

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Metadata
# ---------------------------------------------------------------------------

def get_task_metadata() -> dict:
    return {
        "task_id":            "logreg_lvl1_binary_raw",
        "series":             "Logistic Regression",
        "level":              1,
        "algorithm":          "Logistic Regression (Binary, Raw Tensors)",
        "interface_protocol": "pytorch_task_v1",
    }


# ---------------------------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all relevant RNG seeds for determinism."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 3. Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 4. Data
# ---------------------------------------------------------------------------

def make_dataloaders(
    n_samples: int = 600,
    n_features: int = 2,
    cluster_sep: float = 3.0,
    val_frac: float = 0.2,
    batch_size: int = 64,
    seed: int = 42,
):
    """
    Two Gaussian clusters, standardized features.

    Class 0: X ~ N(-cluster_sep/2, I)
    Class 1: X ~ N(+cluster_sep/2, I)

    Returns train_loader, val_loader (TensorDataset DataLoaders).
    Features are z-score standardized using train statistics only.
    """
    set_seed(seed)
    device = get_device()

    n_per_class = n_samples // 2

    X0 = torch.randn(n_per_class, n_features) - cluster_sep / 2
    X1 = torch.randn(n_per_class, n_features) + cluster_sep / 2

    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([
        torch.zeros(n_per_class, dtype=torch.float32),
        torch.ones(n_per_class,  dtype=torch.float32),
    ], dim=0).unsqueeze(1)  # [N, 1]

    # Shuffle
    perm = torch.randperm(X.size(0))
    X, y = X[perm], y[perm]

    n_val   = int(n_samples * val_frac)
    n_train = n_samples - n_val

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    # Standardize using training statistics
    mean = X_train.mean(dim=0)
    std  = X_train.std(dim=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val,   y_val   = X_val.to(device),   y_val.to(device)

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds   = torch.utils.data.TensorDataset(X_val,   y_val)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 5. Model  (plain tensors — weight vector + bias scalar)
# ---------------------------------------------------------------------------

def build_model(n_features: int = 2, device: torch.device = None) -> dict:
    """
    Returns a dict holding raw parameter tensors.
    No torch.nn.Module; no autograd — requires_grad is False.
    """
    if device is None:
        device = get_device()

    w = torch.zeros(n_features, 1, dtype=torch.float32, device=device)
    b = torch.zeros(1,          dtype=torch.float32, device=device)
    return {"w": w, "b": b}


# ---------------------------------------------------------------------------
# Helper: sigmoid, loss, grad
# ---------------------------------------------------------------------------

def _sigmoid(z: torch.Tensor) -> torch.Tensor:
    """sigma(z) = 1 / (1 + exp(-z))  — numerically stable via clamp."""
    return 1.0 / (1.0 + torch.exp(-z.clamp(-50.0, 50.0)))


def _log_loss(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """Binary cross-entropy: -(1/N)*sum[y*log(p) + (1-y)*log(1-p)]."""
    eps  = 1e-8
    loss = -(y * torch.log(y_hat + eps) + (1 - y) * torch.log(1 - y_hat + eps))
    return loss.mean().item()


def _manual_grads(X: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor):
    """
    dJ/dw = (1/N) * X^T @ (y_hat - y)
    dJ/db = (1/N) * sum(y_hat - y)
    """
    N     = float(X.size(0))
    delta = y_hat - y            # [N, 1]
    dw    = (X.t() @ delta) / N  # [n_features, 1]
    db    = delta.mean()         # scalar
    return dw, db


# ---------------------------------------------------------------------------
# 6. Train  (full-batch gradient descent over epochs)
# ---------------------------------------------------------------------------

def train(
    model: dict,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epochs: int = 500,
    lr: float = 0.5,
) -> dict:
    """
    Gradient descent with manual sigmoid and manual gradients.
    No torch.optim, no loss.backward(), no autograd.

    Accumulates full-batch gradients across mini-batches each epoch,
    then performs a single weight update — equivalent to full-batch GD.

    Returns dict with loss_history and val_loss_history.
    """
    w = model["w"]
    b = model["b"]

    loss_history     = []
    val_loss_history = []

    for _epoch in range(n_epochs):
        # Accumulate full-batch gradient
        total_dw  = torch.zeros_like(w)
        total_db  = torch.zeros(1, device=w.device, dtype=w.dtype)
        epoch_loss = 0.0
        total_n    = 0

        for X_batch, y_batch in train_loader:
            n     = X_batch.size(0)
            z     = X_batch @ w + b
            y_hat = _sigmoid(z)

            dw, db = _manual_grads(X_batch, y_batch, y_hat)

            total_dw   += dw * n
            total_db   += db * n
            epoch_loss += _log_loss(y_hat, y_batch) * n
            total_n    += n

        # Full-dataset average gradient update — NO autograd
        w -= lr * (total_dw / total_n)
        b -= lr * (total_db / total_n)

        val_loss = _eval_loss(w, b, val_loader)
        loss_history.append(epoch_loss / total_n)
        val_loss_history.append(val_loss)

    model["w"] = w
    model["b"] = b

    return {
        "loss_history":     loss_history,
        "val_loss_history": val_loss_history,
    }


def _eval_loss(w, b, loader) -> float:
    """Log-loss over a DataLoader without touching params."""
    total_loss = 0.0
    total_n    = 0
    for X_batch, y_batch in loader:
        y_hat       = _sigmoid(X_batch @ w + b)
        total_loss += _log_loss(y_hat, y_batch) * X_batch.size(0)
        total_n    += X_batch.size(0)
    return total_loss / total_n


# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------

def evaluate(
    model: dict,
    loader: torch.utils.data.DataLoader,
    threshold: float = 0.5,
    split_name: str = "val",
) -> dict:
    """
    Compute log-loss, accuracy, precision, recall, F1, and confusion counts.
    Threshold at 0.5 for binary classification.
    """
    w = model["w"]
    b = model["b"]

    all_y_hat = []
    all_y     = []

    for X_batch, y_batch in loader:
        all_y_hat.append(_sigmoid(X_batch @ w + b))
        all_y.append(y_batch)

    y_hat_all = torch.cat(all_y_hat, dim=0)
    y_all     = torch.cat(all_y,     dim=0)

    loss      = _log_loss(y_hat_all, y_all)
    preds     = (y_hat_all >= threshold).float()
    accuracy  = (preds == y_all).float().mean().item()

    tp = ((preds == 1) & (y_all == 1)).sum().item()
    tn = ((preds == 0) & (y_all == 0)).sum().item()
    fp = ((preds == 1) & (y_all == 0)).sum().item()
    fn = ((preds == 0) & (y_all == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "split":     split_name,
        "loss":      loss,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ---------------------------------------------------------------------------
# 8. Predict
# ---------------------------------------------------------------------------

def predict(
    model: dict,
    X: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Returns binary class predictions (0 or 1) for input tensor X."""
    device = model["w"].device
    z      = X.to(device) @ model["w"] + model["b"]
    y_hat  = _sigmoid(z)
    return (y_hat >= threshold).float().squeeze()


# ---------------------------------------------------------------------------
# 9. Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(history: dict, output_dir: str = ".") -> None:
    """
    Saves:
        logreg_lvl1_loss.png  — train vs validation log-loss curve
    """
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(history["loss_history"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["loss_history"],     label="Train Loss", linewidth=1.5)
    ax.plot(epochs, history["val_loss_history"], label="Val Loss",   linewidth=1.5,
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("Logistic Regression (Binary, Raw Tensors) — Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "logreg_lvl1_loss.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    CFG = {
        "n_samples":    600,
        "n_features":   2,
        "cluster_sep":  3.0,    # separation large enough for >0.90 accuracy
        "val_frac":     0.2,
        "batch_size":   64,
        "n_epochs":     500,
        "lr":           0.5,
        "threshold":    0.5,
        "output_dir":   ".",
        # Quality thresholds
        "min_accuracy": 0.90,
        "max_loss":     0.35,
    }

    print("=" * 60)
    print(f"Task  : {get_task_metadata()['task_id']}")
    print(f"Device: {device}")
    print("=" * 60)

    # ---- Data ----
    train_loader, val_loader = make_dataloaders(
        n_samples   = CFG["n_samples"],
        n_features  = CFG["n_features"],
        cluster_sep = CFG["cluster_sep"],
        val_frac    = CFG["val_frac"],
        batch_size  = CFG["batch_size"],
        seed        = 42,
    )

    # ---- Model ----
    model = build_model(n_features=CFG["n_features"], device=device)

    # ---- Train ----
    print(f"\nTraining for {CFG['n_epochs']} epochs  (lr={CFG['lr']}) ...")
    history = train(
        model,
        train_loader,
        val_loader,
        n_epochs = CFG["n_epochs"],
        lr       = CFG["lr"],
    )

    # ---- Evaluate ----
    train_metrics = evaluate(model, train_loader, threshold=CFG["threshold"], split_name="train")
    val_metrics   = evaluate(model, val_loader,   threshold=CFG["threshold"], split_name="val")

    print("\n--- Train Metrics ---")
    print(f"  Loss      : {train_metrics['loss']:.6f}")
    print(f"  Accuracy  : {train_metrics['accuracy']:.4f}")
    print(f"  Precision : {train_metrics['precision']:.4f}")
    print(f"  Recall    : {train_metrics['recall']:.4f}")
    print(f"  F1        : {train_metrics['f1']:.4f}")

    print("\n--- Validation Metrics ---")
    print(f"  Loss      : {val_metrics['loss']:.6f}")
    print(f"  Accuracy  : {val_metrics['accuracy']:.4f}")
    print(f"  Precision : {val_metrics['precision']:.4f}")
    print(f"  Recall    : {val_metrics['recall']:.4f}")
    print(f"  F1        : {val_metrics['f1']:.4f}")
    print(f"  Confusion : TP={val_metrics['tp']}  TN={val_metrics['tn']}  "
          f"FP={val_metrics['fp']}  FN={val_metrics['fn']}")

    # ---- Learned parameters ----
    w_learned = model["w"].cpu().squeeze().tolist()
    b_learned = model["b"].cpu().item()
    if isinstance(w_learned, float):
        w_learned = [w_learned]
    print(f"\n--- Learned Parameters ---")
    print(f"  w = {[round(v, 4) for v in w_learned]}")
    print(f"  b = {b_learned:.4f}")

    # ---- Save artifacts ----
    print("\n--- Saving Artifacts ---")
    save_artifacts(history, output_dir=CFG["output_dir"])

    # ---- Loss convergence check (smoothed segment comparison) ----
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

    # ---- Quality assertions ----
    print("\n--- Quality Assertions ---")
    try:
        assert val_metrics["accuracy"] > CFG["min_accuracy"], (
            f"Val accuracy {val_metrics['accuracy']:.4f} <= threshold {CFG['min_accuracy']}"
        )
        print(f"  [PASS] Val accuracy {val_metrics['accuracy']:.4f} > {CFG['min_accuracy']}")

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
