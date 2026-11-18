"""Cek cepat: apakah Kafka, Redis, dan Postgres bisa dihubungi dari laptop?

Jalankan setelah `docker compose up -d`:
    python scripts/check_infra.py
"""

import sys
import time
import uuid

OK, FAIL = "✅", "❌"
results = []


def check(name):
    def wrap(fn):
        try:
            detail = fn()
            results.append((OK, name, detail))
        except Exception as e:  # noqa: BLE001 — tujuan skrip ini memang menampilkan semua kegagalan
            results.append((FAIL, name, f"{type(e).__name__}: {e}"))
        return fn
    return wrap


@check("Redis")
def _redis():
    import redis

    r = redis.Redis(host="localhost", port=6379, socket_timeout=3)
    r.ping()
    aof = r.config_get("appendonly")["appendonly"]
    return f"ping ok, AOF={aof}"


@check("Postgres")
def _postgres():
    import psycopg

    with psycopg.connect("postgresql://fraud:fraud@localhost:5433/fraud", connect_timeout=3) as conn:
        version = conn.execute("select version()").fetchone()[0]
    return version.split(",")[0]


@check("Kafka (kirim lalu baca 1 pesan)")
def _kafka():
    from confluent_kafka import Consumer, Producer
    from confluent_kafka.admin import AdminClient, NewTopic

    brokers = "localhost:19092"
    quiet = {"bootstrap.servers": brokers, "log_level": 0}  # sembunyikan log ulang-sambung librdkafka
    topic = "_healthcheck"
    admin = AdminClient(quiet)
    if topic not in admin.list_topics(timeout=5).topics:
        admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])[topic].result(10)

    payload = str(uuid.uuid4())
    p = Producer(quiet)
    p.produce(topic, value=payload)
    p.flush(5)

    c = Consumer({**quiet, "group.id": f"check-{payload}",
                  "auto.offset.reset": "earliest"})
    c.subscribe([topic])
    deadline = time.time() + 15
    try:
        while time.time() < deadline:
            msg = c.poll(1.0)
            if msg is not None and not msg.error() and msg.value().decode() == payload:
                return "pesan terkirim dan terbaca"
    finally:
        c.close()
    raise TimeoutError("pesan tidak terbaca dalam 15 detik")


for status, name, detail in results:
    print(f"{status} {name}: {detail}")
ok = all(s == OK for s, _, _ in results)
if not ok:
    print("\nPetunjuk: pastikan Docker Desktop menyala, lalu jalankan `docker compose up -d`"
          " dan tunggu sampai `docker compose ps` menunjukkan semua layanan (healthy).")
sys.exit(0 if ok else 1)
