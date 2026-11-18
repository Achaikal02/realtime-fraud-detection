"""Isi ulang riwayat fitur di Redis dari transaksi yang tersimpan di Postgres.

Kapan dipakai: Redis sempat mati (transaksi diproses `degraded`, riwayatnya tidak
tercatat) atau datanya hilang/rusak.

Prosedur (keputusan Fase 1: jalur streaming berhenti selama backfill):
    1. Hentikan SEMUA consumer (Ctrl+C). Pesan baru menumpuk aman di Kafka.
    2. python -m fraud.backfill
    3. Nyalakan consumer lagi: ia melanjutkan dari offset terakhir.

Yang dimuat ulang: step S-23 sampai S, dengan S = step terakhir di Postgres.
24 step = jendela fitur terpanjang (recv_*_24h). Semua fitur harus benar, bukan hanya
yang paling berpengaruh: pengaruh fitur bisa berubah setelah model dilatih ulang.

Riwayat diisi dengan fungsi yang SAMA dengan consumer (OnlineFeatureStore), dengan urutan
kedatangan yang sama (step, lalu nomor baris), supaya tidak ada training-serving skew.
"""

import argparse
import time

from fraud.online_features import RECV_WINDOW, OnlineFeatureStore

FEATURE_KEY_PATTERNS = ["amt:*", "recv:*"]


def clear_feature_keys(client) -> int:
    """Hapus hanya key riwayat fitur (bukan seluruh Redis)."""
    deleted = 0
    for pattern in FEATURE_KEY_PATTERNS:
        batch = []
        for key in client.scan_iter(match=pattern, count=1000):
            batch.append(key)
            if len(batch) >= 1000:
                deleted += client.unlink(*batch)
                batch = []
        if batch:
            deleted += client.unlink(*batch)
    return deleted


def replay_into(store: OnlineFeatureStore, txs) -> int:
    """Masukkan transaksi ke riwayat Redis, berurutan. Fitur yang dihitung dibuang."""
    n = 0
    for tx in txs:
        _, degraded = store.update_and_get(tx)
        if degraded:
            raise RuntimeError("Redis tidak bisa dihubungi saat backfill. Pastikan Redis menyala.")
        n += 1
    return n


def load_window(conn, window: int = RECV_WINDOW):
    last = conn.execute("SELECT max(step) FROM transactions").fetchone()[0]
    if last is None:
        raise SystemExit("Tabel transactions kosong; tidak ada yang bisa di-backfill.")
    first = last - window + 1
    rows = conn.execute(
        "SELECT tx_id, step, amount, name_dest FROM transactions WHERE step >= %s "
        "ORDER BY step, CAST(substring(tx_id FROM 4) AS BIGINT)", [first]).fetchall()
    txs = ({"tx_id": r[0], "step": r[1], "amount": r[2], "nameDest": r[3]} for r in rows)
    return first, last, len(rows), txs


def consumer_group_is_active() -> bool | None:
    """True kalau masih ada consumer yang berjalan; None kalau tidak bisa dicek."""
    try:
        from confluent_kafka.admin import AdminClient

        from fraud.consumer import GROUP_ID
        from fraud.stream import BROKERS

        desc = AdminClient({"bootstrap.servers": BROKERS}).describe_consumer_groups([GROUP_ID])
        return len(desc[GROUP_ID].result(10).members) > 0
    except Exception:  # noqa: BLE001 — grup belum ada atau Kafka mati: tidak bisa dipastikan
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=RECV_WINDOW, help="jumlah step yang dimuat ulang")
    p.add_argument("--force", action="store_true", help="jalan walau consumer terdeteksi aktif")
    args = p.parse_args()

    active = consumer_group_is_active()
    if active and not args.force:
        raise SystemExit("Consumer masih berjalan. Hentikan semua consumer dulu (Ctrl+C), "
                         "supaya riwayat tidak tercampur dengan transaksi baru.")
    if active is None:
        print("Peringatan: status consumer tidak bisa dicek. Pastikan semua consumer sudah berhenti.")

    from fraud.db import connect

    store = OnlineFeatureStore.connect()
    conn = connect()
    t0 = time.perf_counter()
    first, last, n_rows, txs = load_window(conn, args.window)
    print(f"Memuat ulang step {first}–{last} ({n_rows:,} transaksi) dari Postgres...")
    print(f"Redis: {clear_feature_keys(store.r):,} key riwayat lama dihapus.")
    n = replay_into(store, txs)
    conn.close()
    print(f"Selesai: {n:,} transaksi dimasukkan ke riwayat dalam {time.perf_counter() - t0:,.1f} detik.")
    print("Sekarang nyalakan consumer lagi: ia melanjutkan dari offset terakhir di Kafka.")


if __name__ == "__main__":
    main()
