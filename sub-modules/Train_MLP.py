"""Train SimpleMLP from current and past x samples."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from Model_MLP import SimpleMLP


def load_xy(
    *,
    data_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_npy = data_dir / "x.npy"
    y_npy = data_dir / "y.npy"
    x_mat = data_dir / "x.mat"
    y_mat = data_dir / "y.mat"

    has_npy = x_npy.is_file() and y_npy.is_file()
    has_mat = x_mat.is_file() and y_mat.is_file()

    if has_npy and has_mat:
        raise FileExistsError(
            f"Found both npy and mat pairs in {data_dir}. Keep only one pair: "
            "x.npy/y.npy OR x.mat/y.mat."
        )

    if has_npy:
        x_np = np.asarray(np.load(x_npy), dtype=np.float32).reshape(-1)
        y_np = np.asarray(np.load(y_npy), dtype=np.float32).reshape(-1)
    elif has_mat:
        mx = loadmat(x_mat)
        my = loadmat(y_mat)
        if "x" not in mx:
            raise KeyError(f"x.mat must contain variable 'x'; found keys: {sorted(mx.keys())}")
        if "y" not in my:
            raise KeyError(f"y.mat must contain variable 'y'; found keys: {sorted(my.keys())}")
        x_np = np.asarray(mx["x"], dtype=np.float32).reshape(-1)
        y_np = np.asarray(my["y"], dtype=np.float32).reshape(-1)
    else:
        raise FileNotFoundError(
            f"Could not find x.npy/y.npy or x.mat/y.mat under {data_dir}"
        )

    return torch.from_numpy(x_np), torch.from_numpy(y_np)


def make_windows(x: torch.Tensor, y: torch.Tensor, L: int) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("x and y must be 1D tensors")
    Nx = x.numel()
    Ny = y.numel()
    if L < 0:
        raise ValueError("L must not be negative")

    if Nx != Ny + L:
        raise ValueError(
            f"length mismatch for required alignment: expected len(x)==len(y)+L, got len(x)={Nx}, len(y)={Ny}, L={L}"
        )

    # y[n] corresponds to conceptual window ending at stored x[n+L]
    n = torch.arange(0, Ny, device=x.device, dtype=torch.long)
    cols = [x[n + L - lag] for lag in range(L + 1)]
    X = torch.stack(cols, dim=1)
    y_t = y
    
    return X, y_t


class WindowDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, L: int) -> None:
        self.X, self.y = make_windows(x, y, L)

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def train(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    lr: float,
    epochs: int,
) -> list[float]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    losses: list[float] = []
    model.train()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        loss_epoch = epoch_loss / max(n_batches, 1)
        losses.append(loss_epoch)
        done = epoch + 1
        if done % 100 == 0 or done == epochs:
            elapsed = time.perf_counter() - t0
            avg = elapsed / done
            eta = avg * (epochs - done)
            print(
                f"epoch {done}/{epochs}  train_mse={loss_epoch:.6e}  "
                f"elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                flush=True,
            )
    return losses


def main() -> None:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Train a simple MLP with DataLoader")
    p.add_argument("--data-dir", type=Path, default=root, help="folder containing x/y files")
    p.add_argument("--L", type=int, default=4, help="max lag (input dim = L+1)")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument(
        "--num-hidden-layers",
        type=int,
        default=2,
        help="number of hidden Linear+ReLU blocks (0 = linear map (L+1)->1 only)",
    )
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 or negative = use full dataset as one batch",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    x, y = load_xy(data_dir=args.data_dir)
    ds = WindowDataset(x, y, args.L)
    bs = len(ds) if args.batch_size <= 0 else min(args.batch_size, len(ds))
    loader = DataLoader(ds, batch_size=bs, shuffle=(bs < len(ds)), drop_last=False)

    model = SimpleMLP(L=args.L, hidden=args.hidden, num_hidden_layers=args.num_hidden_layers)
    losses = train(model, loader, device=device, lr=args.lr, epochs=args.epochs)

    print(f"final MSE (epoch avg): {losses[-1]:.6e}")
    torch.save(
        {
            "model": model.state_dict(),
            "L": args.L,
            "hidden": args.hidden,
            "num_hidden_layers": args.num_hidden_layers,
        },
        root / "Final_simpleMLP.pt",
    )


if __name__ == "__main__":
    main()
