"""Aturan keputusan berbasis biaya. Dipakai di notebook, API, dan backfill."""

FALSE_POSITIVE_COST = 200_000  # Rupiah: biaya memblokir pengguna asli
REVIEW_HOLD_COST = 50_000      # Rupiah: biaya pengguna asli yang dananya ditahan selama REVIEW
                               # (penyederhanaan: aslinya bergantung lama menunggu)


def decide(score: float, amount: float, block_cost: float = FALSE_POSITIVE_COST,
           review_cost: float = REVIEW_HOLD_COST) -> str:
    """Tangga keputusan (18 Nov).

    1. BLOCK  : p x A > (1 - p) x 200.000  -> kasus sangat jelas, tidak membebani analis
    2. REVIEW : p x A > (1 - p) x 50.000   -> dana ditahan di rekening penerima, masuk antrean
    3. ALLOW  : sisanya

    REVIEW selalu lebih murah daripada BLOCK bagi pengguna asli; BLOCK tetap dipakai untuk
    kasus jelas karena kapasitas analis terbatas (~60 kasus per jam).
    """
    expected_loss = score * amount
    if expected_loss > (1 - score) * block_cost:
        return "BLOCK"
    if expected_loss > (1 - score) * review_cost:
        return "REVIEW"
    return "ALLOW"


def should_block(score: float, amount: float, fp_cost: float = FALSE_POSITIVE_COST) -> bool:
    """BLOCK jika perkiraan kerugian karena meloloskan > perkiraan kerugian karena memblokir."""
    return score * amount > (1 - score) * fp_cost


def total_cost(y_true, amount, blocked, fp_cost: float = FALSE_POSITIVE_COST) -> dict:
    """Biaya total dalam Rupiah untuk sekumpulan keputusan.

    - fraud yang diloloskan  -> rugi sebesar nominal
    - pengguna asli diblokir -> rugi fp_cost
    """
    import numpy as np

    y = np.asarray(y_true).astype(bool)
    a = np.asarray(amount, dtype=float)
    b = np.asarray(blocked).astype(bool)
    missed = a[y & ~b].sum()
    false_blocks = (~y & b).sum()
    return {
        "biaya_total": float(missed + false_blocks * fp_cost),
        "fraud_lolos_rp": float(missed),
        "salah_blokir": int(false_blocks),
        "fraud_tertangkap": int((y & b).sum()),
        "fraud_total": int(y.sum()),
    }


def break_even_amount(fraud_rate: float, fp_cost: float = FALSE_POSITIVE_COST) -> float:
    """Nominal minimal agar BLOCK layak jika satu-satunya info adalah tingkat fraud dasar.

    Dipakai untuk mode `scoring mati`: p × amount > (1 − p) × fp_cost.
    """
    if fraud_rate <= 0:
        return float("inf")
    return (1 - fraud_rate) / fraud_rate * fp_cost
