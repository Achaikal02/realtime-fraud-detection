# Real-Time Fraud Detection Engine

Mesin deteksi fraud real-time: transaksi masuk lewat Kafka, fitur dihitung di Redis, model (XGBoost + Isolation Forest) memberi skor, lalu FastAPI memutuskan ALLOW / REVIEW / BLOCK.

## Status
Fase 1: data dan baseline (EDA PaySim).

## Struktur
```
data/raw/          # dataset mentah (tidak di-commit)
data/processed/    # hasil olahan (parquet)
notebooks/         # eksplorasi: 01_eda_paysim.ipynb
src/fraud/         # kode yang dipakai ulang: fitur, model, aturan keputusan
services/          # layanan Docker (api, consumer, dashboard) — mulai Fase 2
tests/
docs/              # catatan keputusan
```

## Menyiapkan
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # agar `import fraud` bisa dipakai dari mana saja
```
Unduh PaySim dari Kaggle (`ealaxi/paysim1`), lalu letakkan CSV-nya di `data/raw/paysim.csv`.

## Melatih dan mendaftarkan model
```bash
python -m fraud.train            # latih ulang dengan step <= 400, daftar ke MLflow sebagai paysim-fraud@champion
mlflow ui --backend-store-uri sqlite:///mlflow.db
pytest
```

## Infrastruktur (Fase 2)
Butuh Docker Desktop.
```bash
docker compose up -d              # Kafka, Redpanda Console, Redis (AOF), Postgres
docker compose ps                 # tunggu sampai semua (healthy)
python scripts/check_infra.py     # cek koneksi dari laptop
```
| Layanan | Alamat dari laptop |
|---|---|
| Kafka | `localhost:19092` |
| Redpanda Console | http://localhost:8080 |
| Redis | `localhost:6379` |
| Postgres | `postgresql://fraud:fraud@localhost:5433/fraud` |

## Menjalankan pipeline real-time
Docker menyala, model `paysim-fraud@champion` sudah terdaftar. Tiap perintah di terminal sendiri:
```bash
python -m fraud.reset --yes          # (opsional) kosongkan topic, Redis, dan Postgres
python -m fraud.consumer             # 1 sampai 3 consumer, terminal berbeda (3 partisi)
python -m fraud.producer             # putar ulang step 377-743
python -m fraud.evaluate             # setelah selesai: bandingkan keputusan dengan label asli
```
- Topic `transactions`: 3 partisi, key = nominal (2 desimal), sehingga pasangan TRANSFER → CASH_OUT
  bernominal sama diproses berurutan. Pesan tidak memuat label (`isFraud`) maupun `newbalance*`.
- Step 377-400 hanya memanaskan riwayat Redis (`WARMUP`); skor dimulai di step 401, periode di luar data latih.
- Consumer: fitur online di Redis (paritas dengan versi batch diuji di `tests/`), skor dengan model yang dimuat
  sekali dari MLflow, keputusan dengan aturan biaya, simpan ke Postgres, BARU catat offset Kafka (at-least-once;
  duplikat ditolak di Redis dan Postgres).
- Redis bermasalah → fitur riwayat `NaN`, `degraded=true`, transaksi tetap dinilai.

## Uji kekacauan: Redis mati di tengah replay
Membuktikan bahwa sistem tetap jalan saat Redis mati, lalu pulih tanpa kehilangan riwayat.
1. `python -m fraud.reset --yes`, lalu jalankan consumer dan producer seperti biasa.
2. Di tengah replay: `docker compose stop redis` → consumer tetap jalan, hitungan `degraded` naik
   (fitur riwayat `NaN`, transaksi tetap dinilai).
3. Sekitar satu menit kemudian: `docker compose start redis` → `degraded` berhenti naik, tetapi riwayat
   selama Redis mati bolong.
4. Ctrl+C consumer (producer boleh terus jalan: pesan menumpuk aman di Kafka).
5. `python -m fraud.backfill` → riwayat 24 step terakhir diisi ulang dari Postgres dengan fungsi fitur yang sama.
6. Jalankan consumer lagi → ia mengejar pesan yang menumpuk dari offset terakhir.
7. `python -m fraud.evaluate` → lihat jumlah transaksi `degraded` dan dampaknya pada biaya.

Backfill memuat 24 step (jendela fitur terpanjang), bukan hanya 2 step milik fitur terpenting:
dengan 2 step, `recv_count_24h` dan `recv_sum_24h` salah di semua transaksi berikutnya (`tests/test_backfill.py`).

## Aturan keputusan
BLOCK jika `skor × nominal > (1 − skor) × Rp 200.000`.
Biaya kesalahan: pengguna asli terblokir = Rp 200.000; fraud lolos = sebesar nominal.
