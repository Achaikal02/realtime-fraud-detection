"""Penyimpanan permanen di Postgres: transaksi mentah + keputusan, satu baris per transaksi."""

import json
import math
import os

POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5433/fraud")

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    tx_id            TEXT PRIMARY KEY,
    step             INTEGER NOT NULL,
    type             TEXT NOT NULL,
    amount           DOUBLE PRECISION NOT NULL,
    name_orig        TEXT,
    oldbalance_org   DOUBLE PRECISION,
    name_dest        TEXT,
    oldbalance_dest  DOUBLE PRECISION,
    score            DOUBLE PRECISION,          -- NULL untuk tipe yang tidak dinilai dan WARMUP
    decision         TEXT NOT NULL,             -- ALLOW / REVIEW / BLOCK / WARMUP
    degraded         BOOLEAN NOT NULL,          -- fitur riwayat kosong karena Redis bermasalah
    features         JSONB,                     -- fitur yang dipakai: untuk audit dan uji paritas
    kafka_partition  INTEGER,
    kafka_offset     BIGINT,
    processed_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- tabel lama (sebelum REVIEW) belum punya kolom ini -> tambahkan tanpa menghapus data
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS review_priority DOUBLE PRECISION;
CREATE INDEX IF NOT EXISTS transactions_step_idx ON transactions (step);
CREATE INDEX IF NOT EXISTS transactions_decision_idx ON transactions (decision);
"""

COLUMNS = ["tx_id", "step", "type", "amount", "name_orig", "oldbalance_org", "name_dest",
           "oldbalance_dest", "score", "decision", "review_priority", "degraded", "features",
           "kafka_partition", "kafka_offset"]

# ON CONFLICT DO NOTHING: transaksi yang diproses ulang (setelah consumer restart)
# tidak menghasilkan baris ganda. Ini membuat "minimal sekali" dari Kafka aman.
INSERT = (f"INSERT INTO transactions ({', '.join(COLUMNS)}) "
          f"VALUES ({', '.join(['%s'] * len(COLUMNS))}) ON CONFLICT (tx_id) DO NOTHING")


def connect(dsn: str = POSTGRES_DSN):
    import psycopg

    conn = psycopg.connect(dsn)
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()
    return conn


def features_json(features: dict) -> str:
    # JSON tidak mengenal NaN -> simpan sebagai null
    return json.dumps({k: (None if isinstance(v, float) and math.isnan(v) else v)
                       for k, v in features.items()})


def insert_rows(conn, rows: list[dict]) -> None:
    with conn.cursor() as cur:
        cur.executemany(INSERT, [tuple(r[c] for c in COLUMNS) for r in rows])
    conn.commit()
