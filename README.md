```text
project/
├── train.py                  # Main training entry point
│── main.py
├── config.py
├
├── data/
│   └── dataset.py             # data construction
│
├── models/
│   ├── kan.py                 # Kolmogorov–Arnold Network (B-spline based)
│   └── coupled.py             # Coupled PI-KAN (Φ1, Φ2, Φ3)
│
├── physics/
│   └── smdp_ode.py            # First-principles ODE models 
│
├── utils/
│   ├── seed.py                # Reproducibility utilities
│   ├── normalization.py       # Min–max normalization to [-1, 1]
│   └── metrics.py             # RMSE / MSE / MAE / NRMSE
│
└── README.md

                              ┌──────────────┐
                              │  Φ1: Solar   │
  t, T2, T4, F1, I, Ta,    →  │  Field       │ ──► T2
                              │  (PI-KAN)    │
                              └──────┬───────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │  Φ2: Thermal │
 t, T2, T3, T4, T6, V1, Ta  → │  Tank        │ ──► T3, T4
                              │  (PI-KAN)    │
                              └──────┬───────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │  Φ3: MD      │
  t, T3, T5, T6, T8, V1,    → │  Module      │ ──► T5, T6, T8
                              │  (PI-KAN)    │
                              └──────────────┘
