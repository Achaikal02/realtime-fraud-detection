import numpy as np
import pandas as pd
from fraud.features import prior_window_stats


def brute_force(key, step, value, window):
    count, total = [], []
    for i in range(len(key)):
        c = s = 0
        for j in range(i):  # hanya baris yang datang lebih dulu
            if key[j] == key[i] and step[j] >= step[i] - window + 1:
                c += 1
                s += value[j]
        count.append(c)
        total.append(s)
    return np.array(count), np.array(total)


def test_prior_window_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 400
    step = np.sort(rng.integers(1, 60, n))
    key = rng.choice(["A", "B", "C", "D", "E"], n)
    value = rng.uniform(0, 100, n).round(2)
    c, t = prior_window_stats(pd.Series(key), pd.Series(step), pd.Series(value), window=24)
    bc, bt = brute_force(key, step, value, 24)
    assert (c == bc).all()
    assert np.allclose(t, bt)


def test_first_transaction_has_no_history():
    c, t = prior_window_stats(pd.Series(["A"]), pd.Series([5]), pd.Series([10.0]), window=24)
    assert c[0] == 0 and t[0] == 0
