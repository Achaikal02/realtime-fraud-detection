"""Fitur versi online: dihitung satu per satu saat transaksi datang, dengan riwayat di Redis.

Harus memberi angka yang SAMA dengan `build_features` (versi batch) untuk transaksi yang
sama dan urutan yang sama -> diuji di tests/test_online_features.py.

Struktur di Redis (satuan waktu = step, sama seperti batch):
- amt:{nominal}  sorted set, member = tx_id,            score = step  -> same_amount_2h
- recv:{akun}    sorted set, member = "tx_id|nominal",  score = step  -> recv_count_24h, recv_sum_24h

Per transaksi, dalam SATU transaksi Redis (MULTI/EXEC):
  hitung riwayat -> tambahkan transaksi ini -> buang yang terlalu lama -> perpanjang umur key.
Hitung SEBELUM tambah, supaya transaksi tidak menghitung dirinya sendiri (sama dengan batch).

Kalau Redis bermasalah (keputusan: opsi B): fitur riwayat = NaN dan degraded=True,
transaksi tetap dinilai. Riwayat transaksi itu tidak tercatat -> diisi ulang lewat backfill.
"""

import math
import time

import redis

from fraud.stream import amount_key

SAME_AMOUNT_WINDOW = 2      # step: step ini + 1 step sebelumnya (sama dengan features.py)
RECV_WINDOW = 24            # step: 24 jam terakhir
KEY_TTL_SECONDS = 6 * 3600  # key yang tidak tersentuh lagi hilang sendiri
HISTORY_FEATURES = ["recv_count_24h", "recv_sum_24h", "same_amount_2h"]
NAN = float("nan")


def static_features(tx: dict) -> dict:
    """Fitur yang cukup dihitung dari isi pesan itu sendiri (tidak butuh riwayat)."""
    amount, ob, od = tx["amount"], tx["oldbalanceOrg"], tx["oldbalanceDest"]
    return {
        "is_transfer": 1.0 if tx["type"] == "TRANSFER" else 0.0,
        "amount": amount,
        "log_amount": math.log1p(amount),
        "oldbalanceOrg": ob,
        "oldbalanceDest": od,
        "amount_to_balance": amount / (ob + 1.0),
        "dest_zero_before": 1.0 if od == 0 else 0.0,
        "kuras": 1.0 if abs(amount - ob) < 0.01 else 0.0,
        "hour": float(tx["step"] % 24),
    }


class OnlineFeatureStore:
    def __init__(self, client: redis.Redis, cooldown_seconds: float = 5.0):
        self.r = client
        # Pemutus arus (circuit breaker): setelah Redis gagal, jangan coba lagi selama
        # beberapa detik. Tanpa ini, SETIAP transaksi menunggu timeout dan consumer
        # jadi sangat lambat persis saat sistem sedang bermasalah.
        self.cooldown = cooldown_seconds
        self._down_until = 0.0

    @classmethod
    def connect(cls, host="localhost", port=6379):
        # timeout pendek: Redis yang mati harus ketahuan cepat, bukan menggantung
        return cls(redis.Redis(host=host, port=port, socket_timeout=0.5, socket_connect_timeout=0.5))

    def update_and_get(self, tx: dict) -> tuple[dict, bool]:
        """Kembalikan (fitur riwayat, degraded)."""
        if time.monotonic() < self._down_until:
            return dict.fromkeys(HISTORY_FEATURES, NAN), True

        step, tx_id = int(tx["step"]), tx["tx_id"]
        amt = amount_key(tx["amount"])
        amt_key, recv_key = f"amt:{amt}", f"recv:{tx['nameDest']}"

        try:
            p = self.r.pipeline(transaction=True)
            p.zscore(amt_key, tx_id)                                        # 0 sudah pernah diproses?
            p.zcount(amt_key, step - SAME_AMOUNT_WINDOW + 1, "+inf")        # 1 riwayat nominal
            p.zrangebyscore(recv_key, step - RECV_WINDOW + 1, "+inf")       # 2 riwayat penerima
            p.zadd(amt_key, {tx_id: step})                                  # 3 baru tambahkan
            p.zadd(recv_key, {f"{tx_id}|{amt}": step})                      # 4
            # buang yang sudah jauh di luar jendela (dengan kelonggaran, karena urutan
            # antar-partisi tidak dijamin)
            p.zremrangebyscore(amt_key, "-inf", f"({step - 2 * SAME_AMOUNT_WINDOW}")
            p.zremrangebyscore(recv_key, "-inf", f"({step - 2 * RECV_WINDOW}")
            p.expire(amt_key, KEY_TTL_SECONDS)
            p.expire(recv_key, KEY_TTL_SECONDS)
            already_seen, same_count, recv_members, *_ = p.execute()
        except redis.exceptions.RedisError:
            self._down_until = time.monotonic() + self.cooldown
            return dict.fromkeys(HISTORY_FEATURES, NAN), True

        # Kafka menjamin "minimal sekali" (at-least-once): transaksi yang sama bisa datang
        # lagi setelah consumer restart. ZADD dengan member yang sama tidak menduplikasi,
        # tetapi hitungannya akan ikut menghitung dirinya sendiri -> dikoreksi di sini.
        if already_seen is not None:
            same_count -= 1
        recv_count, recv_sum = 0, 0.0
        for member in recv_members:
            other_id, other_amt = member.decode().rsplit("|", 1)
            if other_id != tx_id:
                recv_count += 1
                recv_sum += float(other_amt)

        return {"recv_count_24h": float(recv_count), "recv_sum_24h": recv_sum,
                "same_amount_2h": float(same_count)}, False
