import pandas as pd

from fraud.decision import decide
from fraud.review import review_cost_summary, simulate_review


def test_decision_ladder():
    assert decide(0.99, 1_000_000) == "BLOCK"     # sangat jelas
    assert decide(0.30, 200_000) == "REVIEW"      # 60.000 > 35.000, tetapi < 140.000
    assert decide(0.01, 10_000) == "ALLOW"
    # transaksi kecil, skor tinggi: tidak layak diblokir, tetapi layak diperiksa
    assert decide(0.9, 15_000) == "REVIEW"      # 13.500 < 20.000, tetapi > 5.000


def test_priority_queue_capacity_and_expiry():
    # 3 kasus di step 1, kapasitas 1 per step, batas 2 step
    cases = pd.DataFrame({"step": [1, 1, 1], "priority": [10.0, 500.0, 50.0]})
    out = simulate_review(cases, capacity=1, hold_steps=2)
    assert out[1] == "reviewed"        # prioritas tertinggi diperiksa di step 1
    assert out[2] == "reviewed"        # berikutnya di step 2
    assert out[0] == "expired"         # step 3: sudah lewat 2 step -> dana dilepas


def test_cost_summary():
    df = pd.DataFrame({"isFraud": [1, 0, 1, 0], "amount": [30_000.0, 80_000.0, 5_000.0, 1.0],
                       "decision": ["REVIEW", "REVIEW", "ALLOW", "BLOCK"]})
    outcome = pd.Series(["reviewed", "expired"], index=[0, 1])
    s = review_cost_summary(df, outcome, review_cost=50_000)
    assert s["tertangkap_review"] == 1 and s["fraud_lolos"] == 1
    assert s["biaya_total"] == 5_000 + 200_000 + 50_000
