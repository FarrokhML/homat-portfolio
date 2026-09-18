#!/usr/bin/env python3
"""مدل پایه.

    pip install -r requirements.txt
    python baseline.py

قاعده: مدل اصلی باید از مدل پایه بهتر باشد. اگر نبود، مسئله با این دیتا
حل نمی‌شود و بهتر است همان را صادقانه بگویی تا اینکه عدد بی‌معنی منتشر کنی.
"""

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", 'tayaee-bank-customer-churn-prediction-full.csv')
TARGET = 'Churn'
NUMERIC = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
]
CATEGORICAL = [
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'InternetService',
    'Contract',
    'PaymentMethod',
]


def build_pipeline():
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("hot", OneHotEncoder(handle_unknown="ignore", max_categories=25)),
        ]), CATEGORICAL),
    ], remainder="drop")
    return Pipeline([("pre", pre),
                     ("model", RandomForestClassifier(n_estimators=300, random_state=0))])


def main():
    from sklearn.metrics import roc_auc_score

    def score(model, X, y):
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X)[:, 1]
            return roc_auc_score(y, probability)
        return roc_auc_score(y, model.predict(X))

    METRIC = "ROC-AUC"

    frame = pd.read_csv(DATA, sep=',')
    frame = frame.dropna(subset=[TARGET])
    if frame.empty:
        print("بعد از حذف سطرهای بدون هدف، چیزی نماند.")
        return

    X = frame[NUMERIC + CATEGORICAL]
    y = frame[TARGET]
    print(f"{len(frame)} سطر · {len(NUMERIC) + len(CATEGORICAL)} ویژگی")
    print(f"توزیع هدف:\n{y.value_counts(normalize=True).head()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0,
        stratify=y if True and y.nunique() < 20 else None)

    dummy = Pipeline([("pre", build_pipeline().named_steps["pre"]),
                      ("model", DummyClassifier(strategy="most_frequent"))])
    dummy.fit(X_train, y_train)
    base = score(dummy, X_test, y_test)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    real = score(pipeline, X_test, y_test)

    print("=" * 46)
    print(f"معیار         : {METRIC}")
    print(f"مدل پایه      : {base:.4f}")
    print(f"مدل           : {real:.4f}")
    lift = real - base
    print(f"بهبود         : {lift:+.4f}")
    print("=" * 46)
    if lift <= 0.01:
        print("مدل عملاً از پایه بهتر نیست. یا ویژگی کم است، یا این دیتا")
        print("جواب این مسئله را ندارد. همین را صادقانه گزارش کن.")
    else:
        print("این دو عدد را در بات ثبت کن:  /result <شماره رادار>")


if __name__ == "__main__":
    main()
