import numpy as np
import pandas as pd

from fraud.producer import load_range, to_message
from fraud.stream import MESSAGE_FIELDS, amount_key, decode, encode


def test_amount_key_groups_like_batch_round2():
    # versi batch mengelompokkan dengan round(2); versi online dengan amount_key.
    amounts = pd.Series([181.0, 181.00, 9839.64, 10_000_000.0, 0.1 + 0.2, 1864.28])
    batch_groups = pd.factorize(amounts.round(2))[0]
    online_groups = pd.factorize(amounts.map(amount_key))[0]
    assert (batch_groups == online_groups).all()
    assert amount_key(181.0) == "181.00"


def test_message_has_no_label_or_post_transaction_balance():
    assert "isFraud" not in MESSAGE_FIELDS
    assert not any(f.startswith("newbalance") for f in MESSAGE_FIELDS)


def test_encode_decode_roundtrip():
    m = {"tx_id": "tx_1", "step": 401, "amount": 181.0, "type": "TRANSFER"}
    assert decode(encode(m)) == m


def test_load_range_uses_global_row_number(tmp_path):
    n = 50
    df = pd.DataFrame({
        "step": np.repeat(np.arange(1, 11), 5).astype("int16"),
        "type": pd.Categorical(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"] * 10),
        "amount": np.arange(n) * 10.0,
        "nameOrig": [f"C{i}" for i in range(n)],
        "oldbalanceOrg": np.zeros(n),
        "newbalanceOrig": np.zeros(n),
        "nameDest": [f"C{i + 100}" for i in range(n)],
        "oldbalanceDest": np.zeros(n),
        "newbalanceDest": np.zeros(n),
        "isFraud": np.zeros(n, dtype="int8"),
    })
    path = tmp_path / "p.parquet"
    df.to_parquet(path, index=False)

    out = load_range(path, start_step=4, end_step=6)
    assert list(out.tx_id) == [f"tx_{i}" for i in range(15, 30)]   # baris 15..29 di file asli
    assert (out.amount.to_numpy() == df.amount.iloc[15:30].to_numpy()).all()
    msg = to_message(next(out.itertuples(index=False)))
    assert set(msg) == set(MESSAGE_FIELDS)
    assert msg["type"] == "TRANSFER"
