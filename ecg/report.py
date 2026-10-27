"""Results table and README figures: results/summary.md and results/figures/.

    uv run python -m ecg.report

Data figures need data/mitdb/; result figures and the table cover whatever results/ holds.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ecg.data import BEFORE, CLASSES, DS1, DS2, FS, load_records
from ecg.models import MODELS

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
SPLITS = {"intra": "intra-patient (random beats)", "inter": "inter-patient (DS1 → DS2)"}
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CLASS_COLORS = dict(zip(CLASSES, SERIES))
CLASS_NAMES = {"N": "normal", "S": "supraventricular", "V": "ventricular", "F": "fusion"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "axes.titlecolor": TEXT, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9.5, "legend.frameon": False,
})


def beat_shapes(ds1) -> None:
    t = (np.arange(ds1.x.shape[1]) - BEFORE) / FS * 1000
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(12, 2.8), sharey=True)
    for ax, (i, c) in zip(axes, enumerate(CLASSES)):
        x = ds1.x[ds1.y == i, :, 0]
        lo, mid, hi = np.percentile(x, [10, 50, 90], axis=0)
        ax.fill_between(t, lo, hi, color=CLASS_COLORS[c], alpha=0.2, linewidth=0)
        ax.plot(t, mid, color=CLASS_COLORS[c], lw=2)
        ax.set(title=f"{c} · {CLASS_NAMES[c]} (n = {len(x):,})", xlabel="ms from R peak")
    axes[0].set_ylabel("normalised MLII")
    fig.tight_layout()
    fig.savefig(FIGURES / "beat_shapes.png", dpi=150, bbox_inches="tight")


def class_counts(ds1, ds2) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    w = 0.38
    pos = np.arange(len(CLASSES))
    for k, (name, ds) in enumerate((("DS1 (train)", ds1), ("DS2 (test)", ds2))):
        n = np.bincount(ds.y, minlength=len(CLASSES))
        bars = ax.bar(pos + (k - 0.5) * w, n, w, color=SERIES[k], label=name)
        ax.bar_label(bars, [f"{v:,}" for v in n], fontsize=8, color=MUTED, padding=2)
    ax.set_yscale("log")
    ax.set_xticks(pos, [f"{c}\n{CLASS_NAMES[c]}" for c in CLASSES])
    ax.set(ylabel="beats (log scale)", title="AAMI classes per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "class_counts.png", dpi=150, bbox_inches="tight")


def results() -> dict:
    out = {}
    for p in RESULTS.glob("*/*.json"):
        r = json.loads(p.read_text())
        out[(r["split"], r["model"])] = r
    return out


def split_gap(res: dict) -> None:
    models = [m for m in MODELS if ("intra", m) in res and ("inter", m) in res]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    w = 0.38
    pos = np.arange(len(models))
    for k, (split, label) in enumerate(SPLITS.items()):
        f1 = [res[(split, m)]["test"]["macro_f1"] for m in models]
        bars = ax.bar(pos + (k - 0.5) * w, f1, w, color=SERIES[k], label=label)
        ax.bar_label(bars, [f"{v:.2f}" for v in f1], fontsize=8, color=MUTED, padding=2)
    ax.set_xticks(pos, models)
    ax.set(ylim=(0, 1.1), ylabel="test macro-F1")
    ax.set_title("Same models, two ways to split the data", pad=24)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "split_gap.png", dpi=150, bbox_inches="tight")


def confusion(res: dict, model: str = "cnn_lstm_rr") -> None:
    keys = [k for k in (("intra", model), ("inter", model)) if k in res]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.8), squeeze=False)
    for ax, key in zip(axes[0], keys):
        cm = np.array(res[key]["test"]["confusion"])
        share = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{share[i, j]:.2f}\n{cm[i, j]:,}", ha="center", va="center",
                        fontsize=8, color="white" if share[i, j] > 0.5 else TEXT)
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.grid(False)
        ax.set(xlabel="predicted", ylabel="true", title=f"{model}, {key[0]}-patient")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion.png", dpi=150, bbox_inches="tight")


def summary(res: dict) -> str:
    """Test-set table, one row per split and model."""
    header = ["split", "model", "accuracy", "macro-F1"]
    header += [f"{c} {m}" for c in CLASSES for m in ("Se", "+P")]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * 2 + "--:|" * (len(header) - 2)]
    for split, name in SPLITS.items():
        for model in MODELS:
            if (split, model) not in res:
                continue
            t = res[(split, model)]["test"]
            cells = [name, f"`{model}`", f"{t['accuracy']:.3f}", f"{t['macro_f1']:.3f}"]
            cells += [f"{t['per_class'][c][m]:.2f}" for c in CLASSES for m in ("se", "ppv")]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    ds1, ds2 = load_records(DS1), load_records(DS2)
    beat_shapes(ds1)
    class_counts(ds1, ds2)
    res = results()
    split_gap(res)
    confusion(res)
    table = summary(res)
    (RESULTS / "summary.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
