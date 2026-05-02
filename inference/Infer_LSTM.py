"""SimpleLSTM inference for saved x_new data and a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
import torch

from Model_LSTM import SimpleLSTM


def load_x_new(data_dir: Path) -> torch.Tensor:
    p_npy = data_dir / "x_new.npy"
    p_mat = data_dir / "x_new.mat"
    has_npy = p_npy.is_file()
    has_mat = p_mat.is_file()
    if has_npy and has_mat:
        raise FileExistsError(f"Found both {p_npy} and {p_mat}; keep only one.")
    if has_npy:
        arr = np.load(p_npy)
    elif has_mat:
        m = loadmat(p_mat)
        if "x" not in m:
            raise KeyError(f"x_new.mat must contain variable 'x'; keys={sorted(m.keys())}")
        arr = m["x"]
    else:
        raise FileNotFoundError(f"Need {p_npy} or {p_mat}")
    a = np.asarray(arr, dtype=np.float32)
    # Accept:
    # - single sequence: (N,), (N,1), (1,N)
    # - batch: (B,N) or (N,B) where N == 1024+L (checked later)
    if a.ndim == 1:
        return torch.from_numpy(a.reshape(-1))
    if a.ndim == 2:
        if a.shape[0] == 1 or a.shape[1] == 1:
            return torch.from_numpy(a.reshape(-1))
        return torch.from_numpy(a)
    raise ValueError(f"x_new must be 1D or 2D, got shape={a.shape}")


def build_inference_windows(x: torch.Tensor, L: int, out_len: int) -> torch.Tensor:
    N = out_len + L

    if x.dim() == 1:
        Nx = x.numel()
        if Nx != N:
            raise ValueError(f"expected len(x_new)=={out_len}+L={N}, got len(x_new)={Nx}, L={L}")
        n = torch.arange(0, out_len, dtype=torch.long, device=x.device)
        cols = [x[n + L - lag] for lag in range(L + 1)]
        return torch.stack(cols, dim=1)  # (out_len, L+1)

    if x.dim() == 2:
        if x.shape[1] == N:
            xb = x
        elif x.shape[0] == N:
            xb = x.t()
        else:
            raise ValueError(f"expected x_new shape (B,{N}) or ({N},B), got {tuple(x.shape)}")

        B = xb.shape[0]
        n = torch.arange(0, out_len, dtype=torch.long, device=x.device)
        cols = [xb[:, n + L - lag] for lag in range(L + 1)]  # (B,out_len)
        Xb = torch.stack(cols, dim=2)  # (B,out_len,L+1)
        return Xb.reshape(B * out_len, L + 1)

    raise ValueError("x_new must be 1D or 2D")


def main() -> None:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Infer SimpleLSTM from x_new to y_predict (1024)")
    p.add_argument("--data-dir", type=Path, default=root, help="folder with x_new.npy or x_new.mat")
    p.add_argument(
        "--pretrained",
        type=Path,
        default=root / "Final_simpleLSTM.pt",
        help="pretrained weights .pt (dict with keys: model, L, hidden, num_layers)",
    )
    p.add_argument("--out-len", type=int, default=1024, help="length of y_predict")
    p.add_argument("--out-npy", type=Path, default=root / "y_predict_LSTM.npy")
    p.add_argument("--out-mat", type=Path, default=None, help="if set, also save this .mat with variable y_predict")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    payload = torch.load(args.pretrained, map_location=device)
    L = int(payload["L"])
    hidden = int(payload["hidden"])
    num_layers = int(payload["num_layers"])

    x_new = load_x_new(args.data_dir)
    X = build_inference_windows(x_new, L=L, out_len=args.out_len)

    model = SimpleLSTM(L=L, num_layers=num_layers, hidden=hidden)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        y_hat = model(X.to(device)).cpu().numpy().astype(np.float32)

    if x_new.dim() == 2 and not (x_new.shape[0] == 1 or x_new.shape[1] == 1):
        N = args.out_len + L
        B = x_new.shape[0] if x_new.shape[1] == N else x_new.shape[1]
        y_hat = y_hat.reshape(B, args.out_len)
    else:
        if y_hat.shape != (args.out_len,):
            raise RuntimeError(f"unexpected output shape {y_hat.shape}")

    np.save(args.out_npy, y_hat)
    print(f"saved {args.out_npy} shape={y_hat.shape}")

    if args.out_mat is not None:
        savemat(args.out_mat, {"y_predict_LSTM": y_hat.reshape(-1, 1)})
        print(f"saved {args.out_mat}")


if __name__ == "__main__":
    main()
