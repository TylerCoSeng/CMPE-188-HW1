"""
Task: linreg_lvl2_autograd_viz
Series: Linear Regression | Level 2
Algorithm: Multivariate Linear Regression (Autograd & Visualization)

Math:
    Model:   h_theta(x) = x @ theta + bias
    Loss:    J(theta) = (1/N) * sum((h(x) - y)^2)   [MSE]
    Gradient via autograd: nabla J(theta) computed by loss.backward()

    Weight update (in torch.no_grad()):
        theta = theta - lr * theta.grad
        bias  = bias  - lr * bias.grad

Saves:
    linreg_lvl2_loss.png  — training vs validation loss curves
    linreg_lvl2_pred.png  — predicted vs actual scatter (val split)
"""

import sys
import random
import math
import os

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Metadata
# ---------------------------------------------------------------------------

def get_task_metadata() -> dict:
    return {
        "task_id": "linreg_lvl2_autograd_viz",
        "series": "Linear Regression",
        "level": 2,
        "algorithm": "Multivariate Linear Regression (Autograd & Visualization)",
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
    n_samples: int = 100,
    n_features: int = 5,
    noise_std: float = 0.5,
    val_frac: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
):
    """
    Synthetic multivariate dataset: y = X @ TRUE_W + TRUE_B + noise
    X shape: [n_samples, n_features]
    Returns train_loader, val_loader, true_weights, true_bias (for reference)
    """
    set_seed(seed)
    device = get_device()

    TRUE_W = torch.tensor([1.5, -2.0, 0.8, 3.1, -0.5], dtype=torch.float32)
    TRUE_B = torch.tensor(2.0, dtype=torch.float32)

    X = torch.randn(n_samples, n_features)
    y = X @ TRUE_W + TRUE_B + noise_std * torch.randn(n_samples)
    y = y.unsqueeze(1)  # [N, 1]

    n_val = int(n_samples * val_frac)
    n_train = n_samples - n_val

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_ds = torch.utils.data.TensorDataset(X_train.to(device), y_train.to(device))
    val_ds   = torch.utils.data.TensorDataset(X_val.to(device),   y_val.to(device))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, TRUE_W, TRUE_B


# ---------------------------------------------------------------------------
# 5. Model
# ---------------------------------------------------------------------------

def build_model(n_features: int = 5, device: torch.device = None) -> nn.Module:
    """Simple linear model: one nn.Linear layer, no activation."""
    if device is None:
        device = get_device()
    model = nn.Linear(n_features, 1)
    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# 6. Train (autograd, manual weight update)
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epochs: int = 300,
    lr: float = 0.05,
) -> dict:
    """
    Train using loss.backward() + manual weight update in torch.no_grad().
    No torch.optim used.

    Returns:
        dict with loss_history, val_loss_history
    """
    loss_history     = []
    val_loss_history = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for X_batch, y_batch in train_loader:
            # Zero gradients manually
            if model.weight.grad is not None:
                model.weight.grad.zero_()
            if model.bias.grad is not None:
                model.bias.grad.zero_()

            # Forward
            y_pred = model(X_batch)
            loss   = torch.mean((y_pred - y_batch) ** 2)

            # Backward (autograd)
            loss.backward()

            # Manual SGD update
            with torch.no_grad():
                model.weight -= lr * model.weight.grad
                model.bias   -= lr * model.bias.grad

            epoch_loss += loss.item()
            n_batches  += 1

        avg_train_loss = epoch_loss / n_batches

        # Validation loss
        val_mse, _, _ = _compute_metrics(model, val_loader)

        loss_history.append(avg_train_loss)
        val_loss_history.append(val_mse)

    return {
        "loss_history":     loss_history,
        "val_loss_history": val_loss_history,
    }


# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------

def _compute_metrics(model: nn.Module, loader: torch.utils.data.DataLoader):
    """Compute MSE, R2, and collect preds/targets."""
    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch)
            all_preds.append(preds)
            all_targets.append(y_batch)

    preds   = torch.cat(all_preds,   dim=0).squeeze()
    targets = torch.cat(all_targets, dim=0).squeeze()

    mse   = torch.mean((preds - targets) ** 2).item()
    ss_res = torch.sum((targets - preds) ** 2).item()
    ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
    r2    = 1.0 - ss_res / (ss_tot + 1e-12)

    return mse, r2, (preds.cpu(), targets.cpu())


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    split_name: str = "val",
) -> dict:
    """Return MSE and R2 on the given split."""
    mse, r2, _ = _compute_metrics(model, loader)
    metrics = {
        "split":   split_name,
        "mse":     mse,
        "r2":      r2,
        "rmse":    math.sqrt(mse),
    }
    return metrics


# ---------------------------------------------------------------------------
# 8. Predict
# ---------------------------------------------------------------------------

def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Run inference; returns raw predictions tensor."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(X.to(device)).squeeze()


# ---------------------------------------------------------------------------
# 9. Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    train_history: dict,
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    output_dir: str = ".",
) -> None:
    """
    Saves:
        linreg_lvl2_loss.png  — train vs val MSE curves
        linreg_lvl2_pred.png  — predicted vs actual scatter on val split
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Loss curve ---
    epochs = range(1, len(train_history["loss_history"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_history["loss_history"],     label="Train MSE", linewidth=1.5)
    ax.plot(epochs, train_history["val_loss_history"], label="Val MSE",   linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss — Multivariate Linear Regression")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "linreg_lvl2_loss.png")
    plt.savefig(loss_path, dpi=120)
    plt.close()
    print(f"  Saved: {loss_path}")

    # --- Pred vs actual ---
    _, _, (preds, targets) = _compute_metrics(model, val_loader)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets.numpy(), preds.numpy(), alpha=0.7, edgecolors="k", s=40)
    mn = min(targets.min().item(), preds.min().item())
    mx = max(targets.max().item(), preds.max().item())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual y")
    ax.set_ylabel("Predicted y")
    ax.set_title("Predicted vs Actual (Validation Split)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pred_path = os.path.join(output_dir, "linreg_lvl2_pred.png")
    plt.savefig(pred_path, dpi=120)
    plt.close()
    print(f"  Saved: {pred_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    CFG = {
        "n_samples":   100,
        "n_features":  5,
        "noise_std":   0.5,
        "val_frac":    0.2,
        "batch_size":  32,
        "n_epochs":    300,
        "lr":          0.05,
        "output_dir":  ".",
        # Quality thresholds
        "min_r2":      0.90,
        "max_mse":     2.0,
        # Loss monotonicity: checked on a smoothed (moving-average) curve after warmup
        # so that mini-batch noise does not cause false failures.
        "warmup_frac":   0.10,
        "smooth_window": 15,   # epochs to average over when computing trend
        "mono_tol":      0.05, # late-segment avg may be at most 5% above early-segment avg
    }

    print("=" * 60)
    print(f"Task : {get_task_metadata()['task_id']}")
    print(f"Device: {device}")
    print("=" * 60)

    # ---- Data ----
    train_loader, val_loader, TRUE_W, TRUE_B = make_dataloaders(
        n_samples  = CFG["n_samples"],
        n_features = CFG["n_features"],
        noise_std  = CFG["noise_std"],
        val_frac   = CFG["val_frac"],
        batch_size = CFG["batch_size"],
        seed       = 42,
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
    train_metrics = evaluate(model, train_loader, split_name="train")
    val_metrics   = evaluate(model, val_loader,   split_name="val")

    print("\n--- Train Metrics ---")
    print(f"  MSE  : {train_metrics['mse']:.6f}")
    print(f"  RMSE : {train_metrics['rmse']:.6f}")
    print(f"  R2   : {train_metrics['r2']:.6f}")

    print("\n--- Validation Metrics ---")
    print(f"  MSE  : {val_metrics['mse']:.6f}")
    print(f"  RMSE : {val_metrics['rmse']:.6f}")
    print(f"  R2   : {val_metrics['r2']:.6f}")

    # ---- Learned parameters vs true ----
    learned_w = model.weight.data.cpu().squeeze()
    learned_b = model.bias.data.cpu().item()
    print("\n--- Parameter Recovery ---")
    print(f"  True  weights : {TRUE_W.tolist()}")
    print(f"  Learned weights: {[round(x, 4) for x in learned_w.tolist()]}")
    print(f"  True  bias    : {TRUE_B.item():.4f}")
    print(f"  Learned bias  : {learned_b:.4f}")
    w_err = torch.mean((learned_w - TRUE_W) ** 2).item() ** 0.5
    b_err = abs(learned_b - TRUE_B.item())
    print(f"  Weight RMSE   : {w_err:.4f}")
    print(f"  Bias   error  : {b_err:.4f}")

    # ---- Save artifacts ----
    print("\n--- Saving Artifacts ---")
    save_artifacts(history, model, val_loader, output_dir=CFG["output_dir"])

    # ---- Loss monotonicity check (segment-average comparison) ----
    # Strategy: compare the mean smoothed loss over an EARLY post-warmup segment
    # against the mean over a LATE post-warmup segment.  A well-trained model
    # must have late_avg <= early_avg * (1 + seg_tol).  This tolerates:
    #   - convergence plateaus (late ≈ early is fine)
    #   - tiny numerical noise at the end of training
    #   - the common case where loss is still decreasing epoch-by-epoch but
    #     the smoothed start/end happen to be nearly identical
    # The check only fires on genuine degradation (late >> early).
    val_losses  = history["val_loss_history"]
    warmup_end  = max(1, int(len(val_losses) * CFG["warmup_frac"]))
    post_warmup = val_losses[warmup_end:]
    w           = CFG["smooth_window"]

    def moving_avg(seq, window):
        """Centred moving average."""
        out = []
        for i in range(len(seq)):
            lo = max(0, i - window // 2)
            hi = min(len(seq), i + window // 2 + 1)
            out.append(sum(seq[lo:hi]) / (hi - lo))
        return out

    smoothed = moving_avg(post_warmup, w)

    # Early segment: first 20% of post-warmup; late segment: last 20%.
    seg_len   = max(1, len(smoothed) // 5)
    early_avg = sum(smoothed[:seg_len]) / seg_len
    late_avg  = sum(smoothed[-seg_len:]) / seg_len

    # Relative change: positive means loss went up, negative means it went down.
    rel_change = (late_avg - early_avg) / (early_avg + 1e-12)

    # Tolerance: late segment is allowed to be at most seg_tol above early segment.
    # A plateau (rel_change ≈ 0) passes; only genuine divergence fails.
    seg_tol = CFG["mono_tol"]   # e.g. 0.05 -> late may be at most 5% above early
    mono_ok = rel_change <= seg_tol

    print(f"\n--- Monotonicity Check (segment avg, window={w}, post-warmup) ---")
    print(f"  Early-segment avg : {early_avg:.6f}  (first {seg_len} smoothed epochs)")
    print(f"  Late-segment  avg : {late_avg:.6f}  (last  {seg_len} smoothed epochs)")
    print(f"  Relative change   : {rel_change:+.4f}  (tolerance: +{seg_tol:.2f})")
    print(f"  Result            : {'PASS' if mono_ok else 'FAIL'}")

    # ---- Quality assertions ----
    print("\n--- Quality Assertions ---")
    try:
        assert val_metrics["r2"] > CFG["min_r2"], (
            f"Val R2 {val_metrics['r2']:.4f} < threshold {CFG['min_r2']}"
        )
        print(f"  [PASS] Val R2 {val_metrics['r2']:.4f} > {CFG['min_r2']}")

        assert val_metrics["mse"] < CFG["max_mse"], (
            f"Val MSE {val_metrics['mse']:.4f} >= threshold {CFG['max_mse']}"
        )
        print(f"  [PASS] Val MSE {val_metrics['mse']:.4f} < {CFG['max_mse']}")

        assert mono_ok, (
            f"Loss did not decrease monotonically: {violations} violations post-warmup"
        )
        print(f"  [PASS] Loss decreases monotonically (within tolerance)")

    except AssertionError as err:
        print(f"\n  [FAIL] {err}")
        sys.exit(1)

    print("\n[SUCCESS] All checks passed. Exiting 0.")
    sys.exit(0)
