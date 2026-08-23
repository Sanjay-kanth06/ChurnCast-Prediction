import pandas as pd

tx = pd.read_csv("data/raw/transactions.csv", usecols=["msno", "transaction_date"])
tx["transaction_date"] = pd.to_datetime(tx["transaction_date"].astype(str), format="%Y%m%d")

total_members = tx["msno"].nunique()
print(f"total unique members: {total_members}")

for cutoff in ["2017-01-31", "2017-02-15", "2017-02-28", "2017-03-10", "2017-03-20"]:
    n = tx.loc[tx["transaction_date"] <= cutoff, "msno"].nunique()
    print(f"cutoff {cutoff}: {n} members have a transaction on/before it ({100*n/total_members:.1f}%)")