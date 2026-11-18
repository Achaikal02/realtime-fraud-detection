import json

import fakeredis
import numpy as np

from fraud.consumer import process_batch
from fraud.online_features import OnlineFeatureStore
from fraud.scoring import Scorer


class FakeBooster:
    feature_names = ["amount", "kuras", "same_amount_2h"]


class FakeModel:
    """Skor 0.9 kalau kuras saldo, selain itu 0.001."""
    def get_booster(self):
        return FakeBooster()

    def predict_proba(self, X):
        assert list(X.columns) == FakeBooster.feature_names       # urutan kolom dari model
        p = np.where(X.kuras.to_numpy() == 1, 0.9, 0.001)
        return np.column_stack([1 - p, p])


def tx(i, step, type_, amount, ob):
    return {"tx_id": f"tx_{i}", "step": step, "type": type_, "amount": amount, "nameOrig": f"C{i}",
            "oldbalanceOrg": ob, "nameDest": "C999", "oldbalanceDest": 0.0}


def test_process_batch_decisions():
    store, scorer = OnlineFeatureStore(fakeredis.FakeRedis()), Scorer(FakeModel())
    txs = [
        tx(0, 399, "TRANSFER", 5000.0, 5000.0),     # pemanasan: tidak dinilai
        tx(1, 401, "TRANSFER", 500_000.0, 500_000.0),  # kuras saldo, skor 0.9 -> BLOCK
        tx(2, 401, "CASH_OUT", 500_000.0, 900_000.0),  # skor 0.001 -> ALLOW
        tx(3, 401, "PAYMENT", 500_000.0, 500_000.0),   # tipe lain: tidak dinilai -> ALLOW
    ]
    rows = process_batch(txs, store, scorer, score_from_step=401)
    assert [r["decision"] for r in rows] == ["WARMUP", "BLOCK", "ALLOW", "ALLOW"]
    assert rows[0]["score"] is None and rows[3]["score"] is None
    assert rows[1]["score"] == 0.9
    # riwayat tetap terisi saat pemanasan: tx_2 melihat tx_1 (step 401) bernominal sama
    assert json.loads(rows[2]["features"])["same_amount_2h"] == 1.0
    assert not any(r["degraded"] for r in rows)


def test_small_amount_allowed_even_with_high_score():
    # 0.9 x Rp 5.000 = Rp 4.500 < 0.1 x Rp 200.000 = Rp 20.000 -> meloloskan lebih murah
    rows = process_batch([tx(1, 401, "TRANSFER", 5000.0, 5000.0)],
                         OnlineFeatureStore(fakeredis.FakeRedis()), Scorer(FakeModel()), 401)
    assert rows[0]["score"] == 0.9 and rows[0]["decision"] == "ALLOW"


def test_process_batch_degraded_keeps_scoring():
    import redis

    store = OnlineFeatureStore(redis.Redis(port=1, socket_timeout=0.2, socket_connect_timeout=0.2))
    rows = process_batch([tx(1, 401, "TRANSFER", 500_000.0, 500_000.0)], store, Scorer(FakeModel()), 401)
    assert rows[0]["degraded"] is True
    assert rows[0]["decision"] == "BLOCK"                        # tetap dinilai (opsi B)
    assert json.loads(rows[0]["features"])["same_amount_2h"] is None   # NaN disimpan sebagai null
