import sys
import os
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Metadata
# ---------------------------------------------------------------------------

def get_task_metadata() -> dict:
    return {
        "task_id":            "linreg_lvl3_regularization_optim",
        "series":             "Linear Regression",
        "level":              3,
        "algorithm":          "Polynomial Regression (Ridge + SGD + GPU)",
        "interface_protocol": "pytorch_task_v1",
    }

# 2. Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 3. Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. Polynomial feature expansion
# ---------------------------------------------------------------------------

def poly_features(x: torch.Tensor, degree: int) -> torch.Tensor:
    x = x.view(-1, 1)
    return torch.cat([x ** d for d in range(1, degree + 1)], dim=1)

# 5. Data
# ---------------------------------------------------------------------------

def make_dataloaders(
    n_samples: int = 200,
    noise_std: float = 0.5,
    x_range: tuple = (-3.0, 3.0),
    degree: int = 3,
    val_frac: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
):
    set_seed(seed)
    device = get_device()

    x = torch.linspace(x_range[0], x_range[1], n_samples)
    perm = torch.randperm(n_samples)
    x = x[perm]
    y = x ** 2 + noise_std * torch.randn(n_samples)

    n_val   = int(n_samples * val_frac)
    n_train = n_samples - n_val

    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    Phi_train = poly_features(x_train, degree)   # [n_train, degree]
    Phi_val   = poly_features(x_val,   degree)   # [n_val,   degree]

    feat_mean = Phi_train.mean(dim=0)
    feat_std  = Phi_train.std(dim=0) + 1e-8
    Phi_train = (Phi_train - feat_mean) / feat_std
    Phi_val   = (Phi_val   - feat_mean) / feat_std

    y_train = y_train.unsqueeze(1)
    y_val   = y_val.unsqueeze(1)

    Phi_train = Phi_train.to(device)
    y_train   = y_train.to(device)
    Phi_val   = Phi_val.to(device)
    y_val     = y_val.to(device)

    train_ds = torch.utils.data.TensorDataset(Phi_train, y_train)
    val_ds   = torch.utils.data.TensorDataset(Phi_val,   y_val)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False)

    sort_idx  = torch.argsort(x)
    x_all_np  = x[sort_idx].numpy()
    y_all_np  = y[sort_idx].numpy()

    return (
        train_loader, val_loader,
        x_all_np, y_all_np,
        feat_mean, feat_std,
    )

# 6. Model
# ---------------------------------------------------------------------------

def build_model(
    degree: int = 3,
    device: torch.device = None,
) -> nn.Module:

    if device is None:
        device = get_device()
    model = nn.Linear(degree, 1).to(device)
    return model

# 7. Train
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epochs: int = 300,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 1e-2,
) -> dict:
    criterion = nn.MSELoss()

    param_groups = [
        {"params": model.weight, "weight_decay": weight_decay},
        {"params": model.bias,   "weight_decay": 0.0},
    ]
    optimizer = optim.SGD(param_groups, lr=lr, momentum=momentum)

    loss_history     = []
    val_loss_history = []

    for _epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        total_n    = 0

        for Phi_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(Phi_batch)
            loss   = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * Phi_batch.size(0)
            total_n    += Phi_batch.size(0)

        train_loss = epoch_loss / total_n
        val_mse, _, _ = _compute_metrics(model, val_loader)

        loss_history.append(train_loss)
        val_loss_history.append(val_mse)

    return {
        "loss_history":     loss_history,
        "val_loss_history": val_loss_history,
    }

# 8. Evaluate
# ---------------------------------------------------------------------------

def _compute_metrics(model: nn.Module, loader: torch.utils.data.DataLoader):
    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for Phi_batch, y_batch in loader:
            all_preds.append(model(Phi_batch))
            all_targets.append(y_batch)

    preds   = torch.cat(all_preds,   dim=0).squeeze()
    targets = torch.cat(all_targets, dim=0).squeeze()

    mse    = torch.mean((preds - targets) ** 2).item()
    ss_res = torch.sum((targets - preds) ** 2).item()
    ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
    r2     = 1.0 - ss_res / (ss_tot + 1e-12)

    return mse, r2, (preds.cpu(), targets.cpu())


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    split_name: str = "val",
) -> dict:
    mse, r2, _ = _compute_metrics(model, loader)
    return {
        "split": split_name,
        "mse":   mse,
        "rmse":  math.sqrt(mse),
        "r2":    r2,
    }

# 9. Predict
# ---------------------------------------------------------------------------

def predict(model: nn.Module, Phi: torch.Tensor) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(Phi.to(device)).squeeze()

# 10. Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    model: nn.Module,
    x_all_np,
    y_all_np,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    degree: int,
    history: dict,
    output_dir: str = ".",
) -> None:

    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    x_grid = torch.linspace(float(x_all_np.min()), float(x_all_np.max()), 400)
    Phi_grid = poly_features(x_grid, degree)
    Phi_grid = (Phi_grid - feat_mean) / feat_std
    Phi_grid = Phi_grid.to(device)

    model.eval()
    with torch.no_grad():
        y_fit = model(Phi_grid).squeeze().cpu().numpy()

    y_true_curve = x_grid.numpy() ** 2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(x_all_np, y_all_np, s=18, alpha=0.5, color="#999999", label="Data")
    ax.plot(x_grid.numpy(), y_true_curve, "g--", linewidth=1.5,
            label="True: $y=x^2$")
    ax.plot(x_grid.numpy(), y_fit, "r-",  linewidth=2.0,
            label=f"Model fit (degree {degree})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Polynomial Ridge Regression — Fit")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    epochs = range(1, len(history["loss_history"]) + 1)
    ax2.plot(epochs, history["loss_history"],     label="Train MSE", linewidth=1.5)
    ax2.plot(epochs, history["val_loss_history"], label="Val MSE",   linewidth=1.5,
             linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Train vs Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "linreg_lvl3_fit.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")

# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    CFG = {
        "n_samples":    200,
        "noise_std":    0.5,
        "x_range":      (-3.0, 3.0),
        "degree":       3,
        "val_frac":     0.2,
        "batch_size":   32,
        "n_epochs":     300,
        "lr":           0.05,
        "momentum":     0.9,
        "weight_decay": 1e-2,
        "output_dir":   ".",
        "min_val_r2":   0.85,
        "max_val_mse":  2.0,
        "overfit_ratio": 2.5,
    }

    print("=" * 60)
    print(f"Task  : {get_task_metadata()['task_id']}")
    print(f"Device: {device}")
    print("=" * 60)

    (
        train_loader, val_loader,
        x_all_np, y_all_np,
        feat_mean, feat_std,
    ) = make_dataloaders(
        n_samples   = CFG["n_samples"],
        noise_std   = CFG["noise_std"],
        x_range     = CFG["x_range"],
        degree      = CFG["degree"],
        val_frac    = CFG["val_frac"],
        batch_size  = CFG["batch_size"],
        seed        = 42,
    )

    model = build_model(degree=CFG["degree"], device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel : nn.Linear({CFG['degree']}, 1)  ({n_params} parameters)")
    print(f"Ridge : lambda = {CFG['weight_decay']}  (SGD weight_decay)")

    print(f"\nTraining for {CFG['n_epochs']} epochs  "
          f"(SGD, lr={CFG['lr']}, momentum={CFG['momentum']}, "
          f"wd={CFG['weight_decay']}) ...")
    history = train(
        model,
        train_loader,
        val_loader,
        n_epochs     = CFG["n_epochs"],
        lr           = CFG["lr"],
        momentum     = CFG["momentum"],
        weight_decay = CFG["weight_decay"],
    )

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

    w = model.weight.data.cpu().squeeze().tolist()
    b = model.bias.data.cpu().item()
    if isinstance(w, float):
        w = [w]
    print(f"\n--- Learned Parameters ---")
    print(f"  w = {[round(v, 4) for v in w]}")
    print(f"  b = {b:.4f}")

    print("\n--- Saving Artifacts ---")
    save_artifacts(
        model, x_all_np, y_all_np,
        feat_mean, feat_std,
        CFG["degree"], history,
        output_dir=CFG["output_dir"],
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

    # Overfit check
    overfit_ratio = val_metrics["mse"] / (train_metrics["mse"] + 1e-12)
    overfit_ok    = overfit_ratio < CFG["overfit_ratio"]
    print(f"\n--- Overfit Check ---")
    print(f"  Train MSE     : {train_metrics['mse']:.6f}")
    print(f"  Val   MSE     : {val_metrics['mse']:.6f}")
    print(f"  Val/Train MSE : {overfit_ratio:.3f}  (max allowed: {CFG['overfit_ratio']})")
    print(f"  Result        : {'PASS' if overfit_ok else 'FAIL'}")

    print("\n--- Quality Assertions ---")
    try:
        assert val_metrics["r2"] > CFG["min_val_r2"], (
            f"Val R2 {val_metrics['r2']:.4f} <= threshold {CFG['min_val_r2']}"
        )
        print(f"  [PASS] Val R2 {val_metrics['r2']:.4f} > {CFG['min_val_r2']}")

        assert val_metrics["mse"] < CFG["max_val_mse"], (
            f"Val MSE {val_metrics['mse']:.4f} >= threshold {CFG['max_val_mse']}"
        )
        print(f"  [PASS] Val MSE {val_metrics['mse']:.4f} < {CFG['max_val_mse']}")

        assert overfit_ok, (
            f"Severe overfit detected: Val/Train MSE ratio = {overfit_ratio:.3f} "
            f">= {CFG['overfit_ratio']}"
        )
        print(f"  [PASS] No severe overfit (Val/Train MSE ratio = {overfit_ratio:.3f})")

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
