
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "outputs/pd_model"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_CODES = {"02", "03", "09"}
OTHER_CODES = {"01", "15", "16", "96"}
KNOWN_CODES = TARGET_CODES | OTHER_CODES

HORIZON = 36
FEATURES = [
    "log_upb",
    "loan_age",
    "prior90",
    "dpd_group",
]


def month_index(s):
    x = pd.to_numeric(s, errors="coerce")
    year = np.floor(x / 100)
    month = x % 100
    valid = (
        x.notna()
        & (x % 1 == 0)
        & year.between(1990, 2030)
        & month.between(1, 12)
    )
    return (year * 12 + month).where(valid)


def normalize_code(s):
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace("", pd.NA)
        .str.zfill(2)
    )


def dpd_bucket(value):
    text = str(value).strip().upper()

    if text == "RA":
        return "RA"

    if text.isdigit():
        n = int(text)
        if n == 0:
            return "current"
        if n == 1:
            return "30_dpd"
        if n == 2:
            return "60_dpd"
        return "90plus_dpd"

    return "other"


# ==================================================
# 1. Load snapshot features and event outcomes
# ==================================================

print("\n=== 1. Loading Data ===")

snap = pd.read_parquet(
    DATA / "24_snapshot_feature_probe.parquet"
)

events = pd.read_parquet(
    DATA / "03_loan_events_2006_2022.parquet",
    columns=[
        "loan_id",
        "first_period",
        "last_period",
        "first_90_period",
        "zero_code",
        "zero_effective",
    ],
)

snap["loan_id"] = snap["loan_id"].astype("string")
events["loan_id"] = events["loan_id"].astype("string")

if events["loan_id"].duplicated().any():
    raise RuntimeError("Duplicate loan IDs in events")

df = snap.merge(
    events,
    on="loan_id",
    how="left",
    validate="many_to_one",
)

if df["first_period"].isna().any():
    raise RuntimeError("Snapshot loans missing event records")

print(f"Loan-snapshot observations: {len(df):,}")


# ==================================================
# 2. Build future 36-month target
# ==================================================

print("\n=== 2. Constructing Target ===")

df["snapshot_m"] = month_index(df["snapshot"])
df["first_m"] = month_index(df["first_period"])
df["last_m"] = month_index(df["last_period"])
df["first90_m"] = month_index(df["first_90_period"])
df["exit_m"] = month_index(df["zero_effective"])
df["exit_code"] = normalize_code(df["zero_code"])

end_m = df["snapshot_m"] + HORIZON

exit_in_window = (
    df["exit_m"].notna()
    & (df["exit_m"] > df["snapshot_m"])
    & (df["exit_m"] <= end_m)
)

target = (
    exit_in_window
    & df["exit_code"].isin(TARGET_CODES)
)

competing = (
    exit_in_window
    & df["exit_code"].isin(OTHER_CODES)
)

bad_metadata = (
    df["exit_m"].notna()
    != df["exit_code"].notna()
) | (
    df["exit_code"].notna()
    & ~df["exit_code"].isin(KNOWN_CODES)
)

no_target_observed = (
    (df["last_m"] >= end_m)
    & ~exit_in_window
    & ~bad_metadata
)

known = target | competing | no_target_observed

df["target_36m"] = np.where(
    target, 1,
    np.where(known, 0, np.nan)
)

print("Known outcomes:", f"{int(known.sum()):,}")
print("Unknown outcomes:", f"{int((~known).sum()):,}")
print("Positive outcomes:", f"{int(target.sum()):,}")

df = df.loc[known].copy()


# ==================================================
# 3. Point-in-time features
# ==================================================

print("\n=== 3. Building Features ===")

df["loan_age"] = (
    df["snapshot_m"] - df["first_m"]
)

df["prior90"] = (
    df["first90_m"].notna()
    & (df["first90_m"] <= df["snapshot_m"])
).astype(int)

df["current_upb"] = pd.to_numeric(
    df["current_upb"], errors="coerce"
)

df["log_upb"] = np.log1p(
    df["current_upb"].where(df["current_upb"] > 0)
)

df["dpd_group"] = df["dpd_raw"].map(dpd_bucket)

valid_features = (
    df["has_exact_record"].fillna(False)
    & df["valid_upb"].fillna(False)
    & df["valid_dpd"].fillna(False)
    & np.isfinite(df["log_upb"])
    & df["loan_age"].notna()
    & (df["loan_age"] >= 0)
)

print("Invalid feature rows:",
      f"{int((~valid_features).sum()):,}")

df = df.loc[valid_features].copy()

if df[FEATURES].isna().any().any():
    raise RuntimeError("Unexpected missing features")

df["target_36m"] = df["target_36m"].astype(int)

print("\nSamples by snapshot:")
print(
    df.groupby("snapshot")["target_36m"]
    .agg(["size", "sum", "mean"])
    .round(5)
    .to_string()
)


# ==================================================
# 4. Time-based split
# ==================================================

train = df.loc[df["snapshot"] == 201001].copy()
test = df.loc[df["snapshot"] == 202001].copy()

if train.empty or test.empty:
    raise RuntimeError("Training or test snapshot missing")

if train["target_36m"].nunique() != 2:
    raise RuntimeError("Training target has only one class")

X_train = train[FEATURES]
y_train = train["target_36m"]

X_test = test[FEATURES]
y_test = test["target_36m"]

print("\n=== 4. Train / Test ===")
print(f"Train: {len(train):,}, positives: {y_train.sum():,}")
print(f"Test:  {len(test):,}, positives: {y_test.sum():,}")


# ==================================================
# 5. Logistic Regression
# ==================================================

numeric = ["log_upb", "loan_age", "prior90"]
categorical = ["dpd_group"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical,
        ),
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            class_weight=None,
        ),
    ),
])

print("\n=== 5. Training Logistic Regression ===")

model.fit(X_train, y_train)

train_pred = model.predict_proba(X_train)[:, 1]
test_pred = model.predict_proba(X_test)[:, 1]


# ==================================================
# 6. Evaluation
# ==================================================

def evaluate(label, y, pred):
    actual_rate = y.mean()
    predicted_rate = pred.mean()

    print(f"\n{label}")
    print(f"Observations:       {len(y):,}")
    print(f"Actual event rate:  {actual_rate:.5f}")
    print(f"Predicted mean PD:  {predicted_rate:.5f}")
    print(f"ROC-AUC:            {roc_auc_score(y, pred):.4f}")
    print(f"PR-AUC (AP):        {average_precision_score(y, pred):.4f}")
    print(f"PR baseline:        {actual_rate:.5f}")
    print(f"Brier Score:        {brier_score_loss(y, pred):.6f}")
    print(f"Log Loss:           {log_loss(y, pred):.6f}")


print("\n=== 6. Model Evaluation ===")

evaluate("TRAIN 2010", y_train, train_pred)
evaluate("TEST 2020", y_test, test_pred)


# ==================================================
# 7. Save model and predictions
# ==================================================

test_output = test[
    [
        "loan_id",
        "snapshot",
        "current_upb",
        "dpd_group",
        "loan_age",
        "prior90",
        "target_36m",
    ]
].copy()

test_output["predicted_pd"] = test_pred

test_output.to_parquet(
    OUT / "25_baseline_test_predictions.parquet",
    index=False,
)

joblib.dump(
    model,
    OUT / "25_logistic_baseline.joblib",
)

print("\n=== 7. Complete ===")
print("Model saved: 25_logistic_baseline.joblib")
print("Predictions saved: 25_baseline_test_predictions.parquet")
print("This is an initial prototype, not a final validated PD model.")
