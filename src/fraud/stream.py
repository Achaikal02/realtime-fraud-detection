"""Hal-hal yang harus SAMA antara producer dan consumer: alamat broker, nama topic,
format pesan, dan cara membuat key.

Keputusan desain topic `transactions` (17 Nov):
- 3 partisi: sampai 3 consumer bisa bekerja bersamaan di satu laptop.
- key = nominal (2 desimal). TRANSFER dan CASH_OUT pasangannya bernominal sama,
  jadi masuk partisi yang sama dan diproses berurutan -> `same_amount_2h` terjaga.
  Konsekuensi: transaksi ke satu akun bisa tersebar ke beberapa partisi, sehingga
  `recv_count_24h` bisa sedikit meleset. Diterima karena model hampir tidak memakainya.
"""

import json
import os

BROKERS = os.getenv("KAFKA_BROKERS", "localhost:19092")
TOPIC_TRANSACTIONS = "transactions"
NUM_PARTITIONS = 3

# Hanya informasi yang ada SEBELUM transaksi disetujui.
# Sengaja TIDAK dikirim: isFraud (label datang belakangan), newbalance* (kebocoran).
MESSAGE_FIELDS = ["tx_id", "step", "type", "amount", "nameOrig", "oldbalanceOrg",
                  "nameDest", "oldbalanceDest"]


def amount_key(amount: float) -> str:
    """Format nominal yang dipakai sebagai key Kafka DAN key Redis `amt:{...}`.
    Harus sama dengan `round(2)` di features.py agar tidak terjadi training-serving skew."""
    return f"{amount:.2f}"


def encode(tx: dict) -> bytes:
    return json.dumps(tx, separators=(",", ":")).encode("utf-8")


def decode(raw: bytes) -> dict:
    return json.loads(raw)


def ensure_topic(admin, name: str = TOPIC_TRANSACTIONS, partitions: int = NUM_PARTITIONS) -> bool:
    """Buat topic kalau belum ada. Kalau sudah ada dengan jumlah partisi berbeda,
    berhenti dengan pesan jelas (partisi tidak bisa dikurangi, dan menambahnya
    mengubah pemetaan key -> partisi)."""
    from confluent_kafka.admin import NewTopic

    topics = admin.list_topics(timeout=10).topics
    if name in topics:
        existing = len(topics[name].partitions)
        if existing != partitions:
            raise RuntimeError(
                f"Topic '{name}' sudah ada dengan {existing} partisi (seharusnya {partitions}). "
                f"Hapus dulu lewat Console (http://localhost:8080) lalu jalankan ulang."
            )
        return False
    admin.create_topics([NewTopic(name, num_partitions=partitions, replication_factor=1)])[name].result(30)
    return True
