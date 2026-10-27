| split | model | accuracy | macro-F1 | N Se | N +P | S Se | S +P | V Se | V +P | F Se | F +P |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| intra-patient (random beats) | `lstm` | 0.953 | 0.732 | 0.97 | 0.98 | 0.53 | 0.94 | 0.92 | 0.85 | 0.79 | 0.26 |
| intra-patient (random beats) | `cnn_lstm` | 0.986 | 0.907 | 0.99 | 1.00 | 0.92 | 0.81 | 0.96 | 0.98 | 0.89 | 0.74 |
| intra-patient (random beats) | `cnn_lstm_rr` | 0.992 | 0.932 | 1.00 | 1.00 | 0.96 | 0.92 | 0.97 | 0.98 | 0.91 | 0.74 |
| inter-patient (DS1 → DS2) | `lstm` | 0.880 | 0.368 | 0.93 | 0.96 | 0.00 | 0.00 | 0.85 | 0.38 | 0.00 | 0.00 |
| inter-patient (DS1 → DS2) | `cnn_lstm` | 0.704 | 0.343 | 0.72 | 0.97 | 0.05 | 0.01 | 0.87 | 0.33 | 0.13 | 0.03 |
| inter-patient (DS1 → DS2) | `cnn_lstm_rr` | 0.916 | 0.471 | 0.96 | 0.98 | 0.10 | 0.20 | 0.90 | 0.57 | 0.10 | 0.07 |
