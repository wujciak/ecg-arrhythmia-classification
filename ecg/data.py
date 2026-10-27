"""MIT-BIH beats, AAMI classes and the two evaluation splits.

Records are read with wfdb from data/mitdb/ (scripts/download_data.sh). Each annotated beat
becomes one sample: a window of the MLII lead around the R peak plus four RR-interval features.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import train_test_split

DATA = Path("data/mitdb")
FS = 360
BEFORE, AFTER = 90, 90  # samples around the R peak: 0.25 s + 0.25 s

# AAMI EC57 classes. Q (paced and unclassifiable beats) is left out, as is common practice.
CLASSES = ("N", "S", "V", "F")
AAMI = {
    **dict.fromkeys("NLRej", "N"),  # normal, bundle branch blocks, escape beats
    **dict.fromkeys("AaJS", "S"),  # supraventricular ectopic
    **dict.fromkeys("VE", "V"),  # ventricular ectopic
    "F": "F",  # fusion of ventricular and normal
}

# Inter-patient split of de Chazal et al. (2004). The four paced records (102, 104, 107, 217)
# are excluded: AAMI recommends it and paced rhythm is trivial to recognise.
DS1 = (101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
       124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230)
DS2 = (100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
       212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234)


@dataclass
class Beats:
    x: np.ndarray  # (n, BEFORE + AFTER, 1) normalised MLII window
    rr: np.ndarray  # (n, 4) RR features
    y: np.ndarray  # (n,) index into CLASSES
    record: np.ndarray  # (n,) record number, i.e. the patient

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "Beats":
        return Beats(self.x[mask], self.rr[mask], self.y[mask], self.record[mask])

    @staticmethod
    def concat(parts: list["Beats"]) -> "Beats":
        return Beats(*(np.concatenate([getattr(p, f) for p in parts])
                       for f in ("x", "rr", "y", "record")))


_BANDPASS = butter(3, [0.5, 40], btype="bandpass", fs=FS, output="sos")


def load_record(record: int) -> Beats:
    rec = wfdb.rdrecord(str(DATA / str(record)))
    ann = wfdb.rdann(str(DATA / str(record)), "atr")
    lead = rec.sig_name.index("MLII")  # channel 0 in every record except 114
    signal = sosfiltfilt(_BANDPASS, rec.p_signal[:, lead])  # baseline wander and noise
    signal = (signal - np.median(signal)) / signal.std()

    # RR intervals from every beat annotation, so that neighbours of excluded beats stay correct
    is_beat = np.isin(ann.symbol, list(AAMI) + ["/", "f", "Q"])
    peaks = ann.sample[is_beat]
    symbols = np.array(ann.symbol)[is_beat]
    rr = np.diff(peaks) / FS
    record_mean = rr.mean()

    xs, feats, ys = [], [], []
    for i in range(1, len(peaks) - 1):
        cls = AAMI.get(symbols[i])
        p = peaks[i]
        if cls is None or p - BEFORE < 0 or p + AFTER > len(signal):
            continue
        pre, post = rr[i - 1], rr[i]
        local = rr[max(0, i - 10):i].mean()  # the last 10 intervals
        xs.append(signal[p - BEFORE:p + AFTER])
        feats.append([pre / record_mean, post / record_mean, pre / local, post / pre])
        ys.append(CLASSES.index(cls))

    n = len(ys)
    return Beats(np.array(xs, dtype=np.float32)[..., None], np.array(feats, dtype=np.float32),
                 np.array(ys), np.full(n, record))


def load_records(records) -> Beats:
    return Beats.concat([load_record(r) for r in records])


def inter_patient(val_records: tuple[int, ...]) -> dict[str, Beats]:
    """Train on DS1 minus `val_records`, validate on those records, test on DS2."""
    ds1 = load_records(DS1)
    val = np.isin(ds1.record, val_records)
    return {"train": ds1.subset(~val), "val": ds1.subset(val), "test": load_records(DS2)}


def intra_patient(seed: int = 42) -> dict[str, Beats]:
    """The original setup: all 44 records pooled, beats split at random (80/10/10).

    Beats of every patient end up in train and test alike, so this measures how well the model
    recognises patients it has already seen. Kept only as a reference for the inter-patient split.
    """
    beats = load_records(DS1 + DS2)
    idx = np.arange(len(beats))
    train, rest = train_test_split(idx, test_size=0.2, stratify=beats.y, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=beats.y[rest], random_state=seed)
    return {k: beats.subset(v) for k, v in (("train", train), ("val", val), ("test", test))}
