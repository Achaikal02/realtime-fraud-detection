"""Memuat model dan memberi skor.

Model dimuat SEKALI saat consumer start (dari MLflow registry, alias `champion`).
Setelah itu consumer tidak bergantung pada MLflow: kalau MLflow mati, scoring jalan terus.
"""

import pathlib

import numpy as np
import pandas as pd

from fraud.decision import FALSE_POSITIVE_COST, REVIEW_HOLD_COST, decide
from fraud.features import MODEL_TYPES

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_URI = "models:/paysim-fraud@champion"


class Scorer:
    def __init__(self, model, fp_cost: float = FALSE_POSITIVE_COST,
                 review_cost: float = REVIEW_HOLD_COST):
        self.model = model
        # Urutan kolom diambil dari model itu sendiri, bukan diketik ulang di sini:
        # kalau model dilatih ulang dengan urutan fitur berbeda, consumer tetap benar.
        self.features = list(model.get_booster().feature_names)
        self.fp_cost = fp_cost
        self.review_cost = review_cost

    @classmethod
    def from_registry(cls, uri: str = MODEL_URI, tracking_uri: str | None = None):
        import mlflow
        import mlflow.xgboost

        mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{ROOT / 'mlflow.db'}")
        return cls(mlflow.xgboost.load_model(uri))

    def score(self, feature_rows: list[dict]) -> np.ndarray:
        """Peluang fraud untuk sekumpulan transaksi sekaligus (lebih cepat daripada satu per satu)."""
        X = pd.DataFrame(feature_rows, columns=self.features).astype("float64")
        return self.model.predict_proba(X)[:, 1]   # predict_proba, BUKAN predict (label 0/1)

    def decide(self, tx_type: str, amount: float, score: float) -> str:
        if tx_type not in MODEL_TYPES:
            return "ALLOW"                          # PAYMENT, CASH_IN, DEBIT: tidak pernah fraud di PaySim
        # tangga BLOCK / REVIEW / ALLOW (decision.decide)
        return decide(score, amount, self.fp_cost, self.review_cost)
