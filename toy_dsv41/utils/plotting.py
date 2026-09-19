"""Save and plot train / val loss curves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def append_history(
    history: List[Dict],
    step: int,
    train_loss: float,
    val_loss: float,
) -> None:
    history.append(
        {"step": step, "train_loss": float(train_loss), "val_loss": float(val_loss)}
    )


def save_history(path: Path, history: List[Dict], meta: Optional[Dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "history": history}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_history(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_loss_curves(
    history: List[Dict],
    out_png: Path,
    title: str = "Toy DeepSeek-V4.1-Flash — train / val loss",
) -> Path:
    """Write a PNG with train and val loss vs step."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for plots. "
            "Install with: pip install matplotlib\n"
            f"Original error: {e}"
        ) from e

    if not history:
        raise ValueError("empty history — nothing to plot")

    steps = [h["step"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    ax.plot(steps, train, label="train loss", color="#1f77b4", linewidth=2)
    ax.plot(steps, val, label="val loss", color="#d62728", linewidth=2, linestyle="--")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    # annotate overfitting if val rises while train falls
    if len(val) >= 3 and val[-1] > min(val) + 0.3 and train[-1] < train[0] - 0.5:
        ax.annotate(
            "overfit risk\n(val↑ train↓)",
            xy=(steps[-1], val[-1]),
            xytext=(steps[len(steps) // 2], max(val) * 0.95),
            arrowprops=dict(arrowstyle="->", color="#555"),
            fontsize=9,
            color="#555",
        )
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def plot_from_file(history_json: Path, out_png: Optional[Path] = None) -> Path:
    blob = load_history(history_json)
    history = blob["history"]
    meta = blob.get("meta", {})
    preset = meta.get("preset", "toy")
    steps = meta.get("steps", "")
    title = f"Toy DSV4.1 — {preset} ({steps} steps) train / val"
    if out_png is None:
        out_png = history_json.with_name("loss_curves.png")
    return plot_loss_curves(history, out_png, title=title)
