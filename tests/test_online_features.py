"""Uji paritas: fitur online (Redis, satu per satu) harus SAMA dengan fitur batch."""

import fakeredis
import numpy as np
import pandas as pd
import pytest
import redis

from fraud.features import build_features
from fraud.online_features import HISTORY_FEATURES, OnlineFeatureStore, static_features


def make_paysim_like(n=600, seed=0):
    rng = np.random.default_rng(seed)
    step = np.sort(rng.integers(1, 60, n))
    types = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"], n)
    # nominal dan penerima sengaja sering berulang supaya fitur riwayat tidak nol semua
    amount = rng.choice([181.0, 250.5, 9839.64, 10_000_000.0, 77.7], n)
    ob = np.where(rng.random(n) < 0.3, amount, rng.uniform(0, 20_000, n).round(2))
    return pd.DataFrame({
        "step": step, "type": types, "amount": amount,
        "nameOrig": [f"C{i}" for i in range(n)], "oldbalanceOrg": ob, "newbalanceOrig": 0.0,
        "nameDest": rng.choice([f"C{9000 + i}" for i in range(15)], n),
        "oldbalanceDest": np.where(rng.random(n) < 0.4, 0.0, 500.0), "newbalanceDest": 0.0,
        "isFraud": 0,
    })


def run_online(df, store):
    rows = []
    for i, r in df.iterrows():
        tx = {"tx_id": f"tx_{i}", **r.to_dict()}
        hist, degraded = store.update_and_get(tx)
        assert not degraded
        rows.append({**static_features(tx), **hist, "type": r.type})
    return pd.DataFrame(rows)


def test_online_matches_batch():
    df = make_paysim_like()
    batch = build_features(df)                         # hanya TRANSFER + CASH_OUT
    online = run_online(df, OnlineFeatureStore(fakeredis.FakeRedis()))
    online = online[online.type.isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)

    assert batch.same_amount_2h.sum() > 0 and batch.recv_count_24h.sum() > 0  # tesnya bermakna
    for col in HISTORY_FEATURES + ["kuras", "amount_to_balance", "dest_zero_before", "log_amount", "is_transfer"]:
        np.testing.assert_allclose(online[col].to_numpy(), batch[col].to_numpy(dtype=float),
                                   rtol=1e-9, err_msg=col)


def test_reprocessing_same_transaction_gives_same_features():
    store = OnlineFeatureStore(fakeredis.FakeRedis())
    a = {"tx_id": "tx_1", "step": 5, "type": "TRANSFER", "amount": 100.0, "oldbalanceOrg": 100.0,
         "nameDest": "C1", "oldbalanceDest": 0.0}
    b = {**a, "tx_id": "tx_2", "type": "CASH_OUT"}
    store.update_and_get(a)
    first, _ = store.update_and_get(b)
    again, _ = store.update_and_get(b)                 # consumer restart -> pesan datang lagi
    assert first == again == {"recv_count_24h": 1.0, "recv_sum_24h": 100.0, "same_amount_2h": 1.0}


def test_redis_down_gives_nan_and_degraded():
    store = OnlineFeatureStore(redis.Redis(port=1, socket_timeout=0.2, socket_connect_timeout=0.2))
    tx = {"tx_id": "tx_1", "step": 5, "type": "TRANSFER", "amount": 100.0, "oldbalanceOrg": 100.0,
          "nameDest": "C1", "oldbalanceDest": 0.0}
    hist, degraded = store.update_and_get(tx)
    assert degraded and all(np.isnan(v) for v in hist.values())
    # pemutus arus: panggilan berikutnya langsung degraded tanpa menunggu timeout
    _, degraded_again = store.update_and_get({**tx, "tx_id": "tx_2"})
    assert degraded_again
