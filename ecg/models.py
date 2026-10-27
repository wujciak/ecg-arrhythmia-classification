"""Model registry. `lstm` and `cnn_lstm` are the architectures of the first version of this
project, unchanged apart from the input size; `cnn_lstm_rr` adds the RR-interval features."""

from collections.abc import Callable
from dataclasses import dataclass

import keras
from keras import layers
from keras.regularizers import l2

from ecg.data import AFTER, BEFORE, CLASSES

WINDOW = BEFORE + AFTER
N_RR = 4


def lstm(neurons: int = 64, dropout: float = 0.3) -> keras.Model:
    beat = keras.Input((WINDOW, 1), name="beat")
    x = layers.LSTM(neurons, return_sequences=True, kernel_regularizer=l2(0.01))(beat)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(neurons * 2, return_sequences=True, kernel_regularizer=l2(0.01))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(neurons, kernel_regularizer=l2(0.01))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(neurons, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    return keras.Model(beat, layers.Dense(len(CLASSES), activation="softmax")(x), name="lstm")


def _cnn_lstm_features(beat):
    x = layers.Conv1D(64, 3, activation="relu")(beat)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.LSTM(200)(x)
    return layers.Dropout(0.3)(x)


def cnn_lstm() -> keras.Model:
    beat = keras.Input((WINDOW, 1), name="beat")
    out = layers.Dense(len(CLASSES), activation="softmax")(_cnn_lstm_features(beat))
    return keras.Model(beat, out, name="cnn_lstm")


def cnn_lstm_rr() -> keras.Model:
    """CNN-LSTM on the beat shape, concatenated with RR features before the classifier.

    Morphology alone barely separates supraventricular beats (S) from normal ones: an S beat
    usually looks normal but comes early. The RR features carry exactly that timing.
    """
    beat = keras.Input((WINDOW, 1), name="beat")
    rr = keras.Input((N_RR,), name="rr")
    x = layers.Concatenate()([_cnn_lstm_features(beat), layers.Dense(16, activation="relu")(rr)])
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(len(CLASSES), activation="softmax")(layers.Dropout(0.3)(x))
    return keras.Model([beat, rr], out, name="cnn_lstm_rr")


@dataclass(frozen=True)
class Spec:
    build: Callable[[], keras.Model]
    uses_rr: bool


MODELS = {
    "lstm": Spec(lstm, uses_rr=False),
    "cnn_lstm": Spec(cnn_lstm, uses_rr=False),
    "cnn_lstm_rr": Spec(cnn_lstm_rr, uses_rr=True),
}
