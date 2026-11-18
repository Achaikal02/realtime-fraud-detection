"""Kosongkan semua jejak replay supaya eksperimen bisa diulang dari nol:
topic Kafka `transactions` + offset consumer group, riwayat di Redis, tabel di Postgres.

    python -m fraud.reset --yes
"""

import argparse
import time

from fraud.consumer import GROUP_ID
from fraud.stream import BROKERS, TOPIC_TRANSACTIONS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yes", action="store_true", help="wajib, sebagai konfirmasi")
    args = p.parse_args()
    if not args.yes:
        raise SystemExit("Ini menghapus topic, riwayat Redis, dan tabel Postgres. Jalankan dengan --yes.")

    from confluent_kafka.admin import AdminClient

    admin = AdminClient({"bootstrap.servers": BROKERS})
    if TOPIC_TRANSACTIONS in admin.list_topics(timeout=10).topics:
        admin.delete_topics([TOPIC_TRANSACTIONS])[TOPIC_TRANSACTIONS].result(30)
        print(f"Kafka: topic '{TOPIC_TRANSACTIONS}' dihapus.")
    try:
        admin.delete_consumer_groups([GROUP_ID])[GROUP_ID].result(30)
        print(f"Kafka: offset grup '{GROUP_ID}' dihapus.")
    except Exception:  # noqa: BLE001 — grup belum pernah ada, atau sudah ikut terhapus
        pass

    import redis

    redis.Redis(host="localhost", port=6379).flushdb()
    print("Redis: riwayat dikosongkan.")

    import psycopg

    from fraud.db import POSTGRES_DSN

    with psycopg.connect(POSTGRES_DSN) as conn:
        conn.execute("DROP TABLE IF EXISTS transactions")
    print("Postgres: tabel transactions dihapus (dibuat ulang otomatis oleh consumer).")
    time.sleep(2)  # beri Kafka waktu menyelesaikan penghapusan sebelum producer membuat topic lagi


if __name__ == "__main__":
    main()
