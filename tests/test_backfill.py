"""Skenario: Redis mati di tengah jalan, lalu dipulihkan dengan backfill.
Setelah backfill, fitur transaksi berikutnya harus SAMA dengan kalau Redis tidak pernah mati."""

import fakeredis
import numpy as np
import redis

from fraud.backfill import clear_feature_keys, replay_into
from fraud.online_features import OnlineFeatureStore


def make_txs(n=3000, steps=80, seed=1):
    rng = np.random.default_rng(seed)
    step = np.sort(rng.integers(1, steps, n))
    return [{"tx_id": f"tx_{i}", "step": int(step[i]),
             "amount": float(rng.choice([181.0, 250.5, 9839.64, 77.7])),
             "nameDest": f"C{int(rng.integers(0, 40))}"} for i in range(n)]


def test_backfill_restores_history_after_outage():
    txs = make_txs()
    split = 2000
    window_start = txs[split - 1]["step"] - 24 + 1
    upcoming = txs[split:split + 300]

    # A. dunia ideal: Redis tidak pernah mati
    clean = OnlineFeatureStore(fakeredis.FakeRedis())
    for tx in txs[:split]:
        clean.update_and_get(tx)
    expected = [clean.update_and_get(tx)[0] for tx in upcoming]

    # B. Redis mati untuk transaksi 1500..1999 (degraded, riwayatnya tidak tercatat)
    server = fakeredis.FakeServer()
    live = OnlineFeatureStore(fakeredis.FakeRedis(server=server))
    for tx in txs[:1500]:
        live.update_and_get(tx)
    down = OnlineFeatureStore(redis.Redis(port=1, socket_timeout=0.1, socket_connect_timeout=0.1))
    assert all(down.update_and_get(tx)[1] for tx in txs[1500:split])

    # tanpa backfill: fitur transaksi berikutnya salah
    no_backfill = OnlineFeatureStore(fakeredis.FakeRedis())
    for tx in txs[:1500]:
        no_backfill.update_and_get(tx)
    wrong = [no_backfill.update_and_get(tx)[0] for tx in upcoming]
    assert wrong != expected

    # backfill: hapus riwayat, isi ulang 24 step terakhir dari "Postgres" (semua transaksi tersimpan)
    clear_feature_keys(live.r)
    replay_into(live, [tx for tx in txs[:split] if tx["step"] >= window_start])
    got = [live.update_and_get(tx)[0] for tx in upcoming]
    assert got == expected


def test_clear_only_removes_feature_keys():
    r = fakeredis.FakeRedis()
    r.zadd("amt:1.00", {"tx_1": 1})
    r.zadd("recv:C1", {"tx_1|1.00": 1})
    r.set("other:key", "tetap ada")
    assert clear_feature_keys(r) == 2
    assert r.get("other:key") == b"tetap ada"
