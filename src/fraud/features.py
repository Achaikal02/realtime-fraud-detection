"""Fitur PaySim versi batch (untuk latih model).

Aturan utama: setiap fitur hanya boleh memakai informasi yang sudah ada
SEBELUM transaksi disetujui. Jadi:
- `newbalanceOrig` dan `newbalanceDest` tidak dipakai (saldo setelah transaksi).
- Fitur riwayat hanya menghitung transaksi yang datang LEBIH DULU.

Urutan kedatangan = urutan baris di CSV PaySim (sudah urut per step, dan di
dalam satu step TRANSFER fraud muncul tepat sebelum CASH_OUT pasangannya).

Nanti layanan real-time menghitung fitur yang sama dari Redis. Kedua versi
harus memberi angka yang sama untuk transaksi yang sama (diuji di tests/).
"""

import numpy as np
import pandas as pd

MODEL_TYPES = ["TRANSFER", "CASH_OUT"]  # fraud hanya ada di dua tipe ini

BASE_FEATURES = [
    "is_transfer",
    "amount",
    "log_amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_balance",
    "dest_zero_before",
    "recv_count_24h",
    "recv_sum_24h",
    "same_amount_2h",
]
OPTIONAL_FEATURES = {"hour": "hour", "kuras": "kuras"}


def prior_window_stats(key: pd.Series, step: pd.Series, value: pd.Series, window: int):
    """Hitung jumlah dan total `value` dari transaksi SEBELUMNYA dengan key yang sama
    dalam `window` step terakhir (step saat ini ikut dihitung, tetapi hanya baris
    yang datang lebih dulu).

    Asumsi: baris sudah dalam urutan kedatangan dan step tidak pernah turun.
    Cara kerja: urutkan per key (stabil, jadi urutan kedatangan tetap), lalu pakai
    searchsorted untuk mencari baris pertama yang masih di dalam jendela waktu.
    """
    codes, _ = pd.factorize(key)
    codes = codes.astype(np.int64)
    s = step.to_numpy(np.int64)
    v = value.to_numpy(np.float64)

    order = np.argsort(codes, kind="stable")
    combined = codes[order] * 10_000 + s[order]  # naik terus: per key, lalu per step
    lo = np.searchsorted(combined, codes[order] * 10_000 + (s[order] - window + 1), side="left")
    pos = np.arange(len(order))

    csum = np.concatenate([[0.0], np.cumsum(v[order])])
    count_sorted = pos - lo
    sum_sorted = csum[pos] - csum[lo]

    count = np.empty(len(order), dtype=np.int64)
    total = np.empty(len(order), dtype=np.float64)
    count[order] = count_sorted
    total[order] = sum_sorted
    return count, total


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Masukan: seluruh data PaySim (semua tipe, urutan asli).
    Keluaran: hanya TRANSFER dan CASH_OUT, dengan kolom fitur + label + step.
    Fitur riwayat dihitung dari SEMUA tipe transaksi sebelum difilter.
    """
    if not df.step.is_monotonic_increasing:
        raise ValueError("Data harus dalam urutan kedatangan (step tidak boleh turun).")

    out = pd.DataFrame(index=df.index)
    out["step"] = df.step
    out["is_transfer"] = (df.type == "TRANSFER").astype(np.int8)
    out["amount"] = df.amount
    out["log_amount"] = np.log1p(df.amount)
    out["oldbalanceOrg"] = df.oldbalanceOrg
    out["oldbalanceDest"] = df.oldbalanceDest
    out["amount_to_balance"] = df.amount / (df.oldbalanceOrg + 1.0)
    out["dest_zero_before"] = (df.oldbalanceDest == 0).astype(np.int8)
    out["kuras"] = ((df.amount - df.oldbalanceOrg).abs() < 0.01).astype(np.int8)
    out["hour"] = (df.step % 24).astype(np.int8)

    # Berapa kali akun penerima ini sudah menerima uang dalam 24 jam terakhir?
    out["recv_count_24h"], out["recv_sum_24h"] = prior_window_stats(
        df.nameDest, df.step, df.amount, window=24
    )
    # Berapa transaksi sebelumnya dengan nominal persis sama dalam 2 jam terakhir?
    # (menangkap pola TRANSFER -> CASH_OUT bernominal sama)
    out["same_amount_2h"], _ = prior_window_stats(
        df.amount.round(2), df.step, df.amount, window=2
    )

    out["isFraud"] = df.isFraud
    return out[df.type.isin(MODEL_TYPES)].reset_index(drop=True)


def temporal_split(feat: pd.DataFrame, max_step: int = 400, fractions=(0.70, 0.15, 0.15)):
    """Pemotongan opsi A: hanya step <= max_step, dibagi berurutan waktu
    berdasarkan jumlah baris. Batas jatuh di pergantian step agar satu jam
    tidak terbelah ke dua set. Step > max_step dikembalikan sebagai set drift.
    """
    main = feat[feat.step <= max_step]
    drift = feat[feat.step > max_step]
    cum = main.groupby("step").size().cumsum() / len(main)
    cut1 = int((cum >= fractions[0]).idxmax())
    cut2 = int((cum >= fractions[0] + fractions[1]).idxmax())
    train = main[main.step <= cut1]
    val = main[(main.step > cut1) & (main.step <= cut2)]
    test = main[main.step > cut2]
    return train, val, test, drift, (cut1, cut2)
