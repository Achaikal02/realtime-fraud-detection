"""Simulasi antrean REVIEW: analis memeriksa kasus dengan perkiraan kerugian terbesar dulu.

Aturan (keputusan 18 Nov):
- Antrean prioritas (opsi B), diurutkan berdasarkan skor x nominal.
- Kapasitas: `capacity` kasus per step (1 step = 1 jam; default 60).
- Selama menunggu, dana DITAHAN di rekening penerima -> penipu tidak bisa mencairkan.
- Kasus yang belum diperiksa setelah `hold_steps` step: dana dilepas (kedaluwarsa).

Hasil per kasus:
- diperiksa + fraud      -> tertangkap, rugi 0
- diperiksa + asli       -> pengguna menunggu, biaya review_cost
- kedaluwarsa + fraud    -> uang hilang sebesar nominal
- kedaluwarsa + asli     -> pengguna sudah menunggu, biaya review_cost
"""

import heapq

import numpy as np
import pandas as pd


def simulate_review(cases: pd.DataFrame, capacity: int = 60, hold_steps: int = 2) -> pd.Series:
    """`cases`: kolom step, priority. Kembalikan Series 'reviewed'/'expired' (index sama)."""
    outcome = pd.Series("expired", index=cases.index, dtype=object)
    if cases.empty:
        return outcome
    by_step = {s: g for s, g in cases.groupby("step")}
    pending = []                                       # heap: (-priority, step_masuk, index)
    for s in range(int(cases.step.min()), int(cases.step.max()) + hold_steps + 1):
        for idx, row in by_step.get(s, pd.DataFrame()).iterrows():
            heapq.heappush(pending, (-row.priority, s, idx))
        # buang yang sudah melewati batas waktu (dana dilepas)
        pending = [c for c in pending if c[1] > s - hold_steps]
        heapq.heapify(pending)
        for _ in range(min(capacity, len(pending))):   # analis mengambil yang teratas
            _, _, idx = heapq.heappop(pending)
            outcome[idx] = "reviewed"
    return outcome


def review_cost_summary(df: pd.DataFrame, outcome: pd.Series, review_cost: float) -> dict:
    """`df`: kolom isFraud, amount, decision. `outcome` untuk baris REVIEW."""
    y, a, d = df.isFraud.astype(bool), df.amount, df.decision
    rev = d == "REVIEW"
    reviewed = rev & (outcome.reindex(df.index) == "reviewed")
    expired = rev & ~reviewed
    lost = a[y & ((d == "ALLOW") | expired)].sum()
    cost = lost + (~y & (d == "BLOCK")).sum() * 200_000 + (~y & rev).sum() * review_cost
    return {
        "biaya_total": float(cost),
        "tertangkap_block": int((y & (d == "BLOCK")).sum()),
        "tertangkap_review": int((y & reviewed).sum()),
        "fraud_lolos": int((y & ((d == "ALLOW") | expired)).sum()),
        "kasus_review": int(rev.sum()),
        "kedaluwarsa": int(expired.sum()),
        "asli_ditahan": int((~y & rev).sum()),
        "salah_blokir": int((~y & (d == "BLOCK")).sum()),
    }
