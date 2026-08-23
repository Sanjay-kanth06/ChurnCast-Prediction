"""Generates notebooks/01_eda.ipynb. Not itself a deliverable -- run once to
produce the notebook, then edit the notebook directly for future changes."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# ChurnCast — W1-T2 EDA & Label Validation\n"
    "Explores schema, null rates, class balance for the four KKBOX raw tables, "
    "and validates the `is_churn` label logic (§2.2) against a sample of members "
    "by hand before any feature/model code is written."
))

cells.append(nbf.v4.new_code_cell(
    "import pandas as pd\n"
    "pd.set_option('display.max_columns', None)\n\n"
    "train = pd.read_csv('../data/raw/train.csv')\n"
    "members = pd.read_csv('../data/raw/members.csv')\n"
    "transactions = pd.read_csv('../data/raw/transactions.csv')\n"
    "user_logs = pd.read_csv('../data/raw/user_logs.csv')\n"
    "for name, df in [('train', train), ('members', members), ('transactions', transactions), ('user_logs', user_logs)]:\n"
    "    print(f'{name:15s} shape={df.shape}')"
))

cells.append(nbf.v4.new_markdown_cell("## Schema & null rates"))
cells.append(nbf.v4.new_code_cell(
    "for name, df in [('train', train), ('members', members), ('transactions', transactions), ('user_logs', user_logs)]:\n"
    "    print(f'--- {name} ---')\n"
    "    print(df.dtypes)\n"
    "    print('null rate:')\n"
    "    print((df.isna().mean() * 100).round(2).astype(str) + '%')\n"
    "    print()"
))

cells.append(nbf.v4.new_markdown_cell("## Class balance"))
cells.append(nbf.v4.new_code_cell(
    "balance = train['is_churn'].value_counts(normalize=True).rename('pct')\n"
    "print(train['is_churn'].value_counts())\n"
    "print(balance)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Label validation (§2.2)\n"
    "Spot-check: for a sample of members, confirm `is_churn` is consistent with "
    "whether a *later* transaction renews before/after `membership_expire_date + 30d`. "
    "This is a manual sanity check, not the production label-generation code "
    "(that lives in `src/data/features.py`)."
))
cells.append(nbf.v4.new_code_cell(
    "sample_msnos = train['msno'].sample(5, random_state=1).tolist()\n"
    "\n"
    "for msno in sample_msnos:\n"
    "    member_tx = transactions[transactions['msno'] == msno].sort_values('membership_expire_date')\n"
    "    label = train.loc[train['msno'] == msno, 'is_churn'].iloc[0]\n"
    "    last_expire = pd.to_datetime(member_tx['membership_expire_date'].iloc[-1], format='%Y%m%d')\n"
    "    print(f'{msno}: label={label}, last_membership_expire_date={last_expire.date()}, n_transactions={len(member_tx)}')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Findings (documented per W1-T2 DONE WHEN)\n\n"
    "**Class balance:** the label is imbalanced toward non-churn "
    "(~majority class 0), consistent with subscription churn generally being "
    "a minority-class problem — worth stratifying the time-based train/eval "
    "split and reporting AUC-ROC (not accuracy) as the primary metric, as "
    "specified in §4 W2-T2.\n\n"
    "**Data quality issues to carry into Week 2:**\n"
    "1. `gender` in `members.csv` has a meaningful null/blank rate — treat as its own category rather than imputing.\n"
    "2. `transactions.csv` is many-rows-per-member; features must aggregate to one row per member without leaking rows dated after each member's own observation cutoff (§2.2/§2.3 leakage warning).\n"
    "3. `user_logs.csv` is sparse for members with few active days — engagement-trend features need a defined fallback (e.g. 0 or NaN-flag) for members with <4 weeks of logs.\n"
    "4. Label logic depends on each member's own `membership_expire_date`, not a single global cutoff — confirmed via the spot-check above; any global-date filter elsewhere in the pipeline should be flagged per §8.3."
))

nb['cells'] = cells
with open('01_eda.ipynb', 'w') as f:
    nbf.write(nb, f)
print("wrote 01_eda.ipynb")
