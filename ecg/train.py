"""Trains one model on one split and evaluates it on the test records.

    uv run python -m ecg.train --model cnn_lstm_rr --split inter
    uv run python -m ecg.train --model lstm --split intra --smoke     # seconds, pipeline check

Outputs: models/<split>_<model>.keras and results/<split>/<model>.json (metrics and history).
Early stopping and the reported epoch are chosen on the validation set; test is evaluated once.
"""

import argparse
import json
import time
from pathlib import Path

import keras
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from ecg.data import CLASSES, Beats, inter_patient, intra_patient
from ecg.models import MODELS

MODELS_DIR = Path("models")
RESULTS = Path("results")
VAL_RECORDS = (201, 207, 223)  # DS1 records with S and V beats, held out for early stopping
PATIENCE = 8


def evaluate(y: np.ndarray, pred: np.ndarray) -> dict:
    """AAMI-style metrics: sensitivity (Se) and positive predictive value (+P) per class."""
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    tp = np.diag(cm)
    se = tp / np.maximum(cm.sum(1), 1)
    ppv = tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(se + ppv > 0, 2 * se * ppv / np.maximum(se + ppv, 1e-12), 0)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "per_class": {c: {"se": float(se[i]), "ppv": float(ppv[i]), "f1": float(f1[i]),
                          "support": int(cm[i].sum())} for i, c in enumerate(CLASSES)},
        "confusion": cm.tolist(),
    }


def inputs(beats: Beats, uses_rr: bool):
    return [beats.x, beats.rr] if uses_rr else beats.x


class MacroF1(keras.callbacks.Callback):
    """Adds val_macro_f1 to the epoch logs so that EarlyStopping can monitor it."""

    def __init__(self, x, y):
        super().__init__()
        self.x, self.y = x, y

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.x, batch_size=1024, verbose=0).argmax(1)
        logs["val_macro_f1"] = evaluate(self.y, pred)["macro_f1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--split", choices=["inter", "intra"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)
    data = inter_patient(VAL_RECORDS) if args.split == "inter" else intra_patient(args.seed)
    if args.smoke:
        rng = np.random.default_rng(0)
        data = {k: v.subset(rng.choice(len(v), 256, replace=False)) for k, v in data.items()}
        args.epochs = 1
    spec = MODELS[args.model]
    for k, v in data.items():
        print(f"{k:5s} {len(v):6d} beats  " + "  ".join(
            f"{c} {n}" for c, n in zip(CLASSES, np.bincount(v.y, minlength=len(CLASSES)))))

    train = data["train"]
    present = np.unique(train.y)
    # square root of the balanced weights: full balancing (F ~30x N) made the models call
    # a fifth of the normal beats F
    weights = np.sqrt(compute_class_weight("balanced", classes=present, y=train.y))
    model = spec.build()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    start = time.time()
    history = model.fit(
        inputs(train, spec.uses_rr), train.y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=dict(zip(present.tolist(), weights)),
        callbacks=[
            MacroF1(inputs(data["val"], spec.uses_rr), data["val"].y),
            keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=PATIENCE,
                                          restore_best_weights=True),
        ],
        verbose=2,
    )

    hist = history.history
    metrics = {
        "model": args.model,
        "split": args.split,
        "best_epoch": int(np.argmax(hist["val_macro_f1"])) + 1,
        "train_seconds": round(time.time() - start),
        "history": {"loss": hist["loss"], "val_macro_f1": hist["val_macro_f1"]},
    }
    for k in ("val", "test"):
        pred = model.predict(inputs(data[k], spec.uses_rr), batch_size=1024, verbose=0).argmax(1)
        metrics[k] = evaluate(data[k].y, pred)

    # smoke runs write only to models/smoke/ so they never overwrite a real run
    model_dir = MODELS_DIR / "smoke" if args.smoke else MODELS_DIR
    result_dir = model_dir if args.smoke else RESULTS / args.split
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / f"{args.split}_{args.model}.keras")
    (result_dir / f"{args.model}.json").write_text(json.dumps(metrics, indent=2) + "\n")

    t = metrics["test"]
    print(f"[{args.split} {args.model}] test macro-F1 {t['macro_f1']:.3f}  " + "  ".join(
        f"{c} Se {m['se']:.2f} +P {m['ppv']:.2f}" for c, m in t["per_class"].items()))


if __name__ == "__main__":
    main()
