"""Latih model final dan daftarkan ke MLflow Model Registry.

Pemakaian (dari folder proyek):
    python -m fraud.train                      # pakai default
    python -m fraud.train --max-step 400 --n-estimators 1000

Perbedaan dengan notebook 02:
- Notebook 02 MEMILIH model (train vs validasi, early stopping).
- Skrip ini MELATIH ULANG model terpilih dengan SEMUA data step <= max-step.
  Karena tidak ada lagi set validasi, early stopping tidak bisa dipakai.
  Jumlah pohon diambil dari hasil notebook 02 (pohon_terpakai sekitar 990).
"""

import argparse
import pathlib

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from fraud.decision import FALSE_POSITIVE_COST, total_cost
from fraud.features import BASE_FEATURES, build_features

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_NAME = "paysim-fraud"
FEATURES = BASE_FEATURES + ["kuras"]  # model default dari notebook 02


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data/processed/paysim.parquet"))
    p.add_argument("--max-step", type=int, default=400)
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--alias", default="champion")
    args = p.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("paysim-final")

    feat = build_features(pd.read_parquet(args.data))
    feat[FEATURES] = feat[FEATURES].astype("float64")
    train = feat[feat.step <= args.max_step]
    drift = feat[feat.step > args.max_step]
    print(f"Latih: {len(train):,} baris ({int(train.isFraud.sum()):,} fraud) | drift: {len(drift):,} baris")

    params = dict(n_estimators=args.n_estimators, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=42)

    with mlflow.start_run(run_name=f"final-step<={args.max_step}") as run:
        model = XGBClassifier(**params).fit(train[FEATURES], train.isFraud)

        # Pemeriksaan di set drift: bukan untuk memilih model, hanya sebagai catatan.
        s = model.predict_proba(drift[FEATURES])[:, 1]
        blocked = s * drift.amount.to_numpy() > (1 - s) * FALSE_POSITIVE_COST
        metrics = {
            "drift_roc_auc": roc_auc_score(drift.isFraud, s),
            "drift_pr_auc": average_precision_score(drift.isFraud, s),
            **{f"drift_{k}": v for k, v in total_cost(drift.isFraud, drift.amount, blocked).items()},
        }
        mlflow.log_params({**params, "max_step": args.max_step, "features": ",".join(FEATURES)})
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"features": FEATURES, "fp_cost": FALSE_POSITIVE_COST}, "model_contract.json")

        info = mlflow.xgboost.log_model(
            model,
            name="model",
            input_example=train[FEATURES].head(5),
            registered_model_name=MODEL_NAME,
        )

    client = MlflowClient()
    version = info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, args.alias, version)
    print(f"\nTerdaftar: {MODEL_NAME} versi {version}, alias '{args.alias}'")
    print(f"Muat dengan: mlflow.xgboost.load_model('models:/{MODEL_NAME}@{args.alias}')")
    for k, v in metrics.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
