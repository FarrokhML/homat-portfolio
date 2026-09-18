#!/usr/bin/env python3
"""pipeline.py — تحلیل قابل بازتولید یک دیتاست رادار هومات.

این فایل را بات «هومات» در پوشه‌ی پروژه گذاشته و <b>همین فایل</b> را اجرا کرده؛
عددهای گزارش و محتوا از خروجی همین اجراست. برای بازتولید:

    pip install pandas numpy scikit-learn matplotlib joblib
    python pipeline.py                 # تنظیمات از lab_config.json
    python pipeline.py --target Churn  # هدف دیگر

خروجی‌ها در پوشه‌ی out/:
    results.json   همه‌ی عددها (پایه، مدل‌ها، بازه‌ی اطمینان، اهمیت ویژگی‌ها)
    figures/*.png  نمودارها
    model.joblib   بهترین مدل، آماده‌ی استفاده روی داده‌ی تازه

روش، به ترتیب:
  1. پاک‌سازی: حذف شناسه، ستون ثابت، ستون بیش از ۶۰٪ خالی، متن آزاد پرتنوع
  2. تاریخ ← سال/ماه/روز هفته/ساعت، و فاصله‌ی روز بین دو ستون تاریخ
  3. غربال نشت داده: ویژگی‌ای که به‌تنهایی هدف را تقریباً کامل پیش‌بینی کند
     کنار گذاشته و گزارش می‌شود (معمولاً ستونی است که بعد از رخداد پر شده)
  4. تقسیم: اگر تاریخ هست، زمانی (۲۰٪ آخر آزمون)؛ وگرنه تصادفی لایه‌بندی‌شده
  5. پایه‌ی ساده (بدون مدل) + سه مدل؛ انتخاب با اعتبارسنجی متقاطع ۵تایی
  6. سنجش روی داده‌ی آزمون که مدل هرگز ندیده، با بازه‌ی اطمینان bootstrap
  7. اهمیت جایگشتی ویژگی‌ها روی داده‌ی آزمون
"""

import argparse
import json
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
MAX_ROWS = 200_000
CV_ROWS = 60_000
# کتابخانه‌هایی که واقعاً در این اجرا کار کردند — در results.json و در بخش
# «منابع» گزارش می‌آیند. ادعای استفاده از کتابخانه‌ای که اجرا نشده، ممنوع است.
USED = {"pandas", "scikit-learn", "matplotlib"}
# «.*id» غلط بود: paid و valid و fluid را هم شناسه می‌دانست. شناسه یا جداکننده
# دارد (customer_id) یا camelCase است (customerID) — نه هر کلمه‌ای که به id ختم شود.
ID_LIKE = re.compile(r"^(id|.*[_\s.-]id|index|idx|uuid|guid|key|row_?num(ber)?|"
                     r"unnamed.*|file_?name|path|url|image.*)$", re.I)
CAMEL_ID = re.compile(r"[a-z0-9](Id|ID)$")


def is_id(name):
    return bool(ID_LIKE.match(name.replace(" ", "_")) or CAMEL_ID.search(name))


# ------------------------------------------------------------------ داده

def is_text(series):
    """متنی یا دسته‌ای؟ در pandas 3 ستون متنی دیگر object نیست، str است."""
    return not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
                or pd.api.types.is_datetime64_any_dtype(series))


def load(path):
    if path.lower().endswith(".parquet"):
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig",
                            nrows=MAX_ROWS, on_bad_lines="skip")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame.head(MAX_ROWS)


def parse_dates(frame, target):
    dates = []
    for col in frame.columns:
        if col == target or not is_text(frame[col]):
            continue
        sample = frame[col].dropna().astype(str).head(200)
        if sample.empty or not sample.str.match(r"^\d{4}-\d{1,2}-\d{1,2}|^\d{1,2}/\d{1,2}/\d{2,4}").mean() > 0.9:
            continue
        parsed = pd.to_datetime(frame[col], errors="coerce", utc=True)
        if parsed.notna().mean() > 0.9:
            frame[col] = parsed.dt.tz_localize(None)
            dates.append(col)
    return dates


def engineer(frame, dates):
    notes = []
    for col in dates:
        d = frame[col]
        frame[f"{col}__month"] = d.dt.month
        frame[f"{col}__dow"] = d.dt.dayofweek
        if (d.dt.hour > 0).mean() > 0.2:
            frame[f"{col}__hour"] = d.dt.hour
        if d.dt.year.nunique() > 1:
            frame[f"{col}__year"] = d.dt.year
    for i, a in enumerate(dates):
        for b in dates[i + 1:]:
            gap = (frame[b] - frame[a]).dt.total_seconds() / 86400
            if gap.abs().median() > 0:
                frame[f"days_{a}_to_{b}"] = gap.round(2)
                notes.append(f"فاصله‌ی روز بین {a} و {b} ساخته شد")
    return frame, notes


def clean(frame, target):
    dropped = {}
    for col in list(frame.columns):
        if col == target:
            continue
        series = frame[col]
        reason = None
        if is_id(col):
            reason = "شناسه"
        elif series.nunique(dropna=True) <= 1:
            reason = "ثابت"
        elif series.isna().mean() > 0.6:
            reason = "بیش از ۶۰٪ خالی"
        elif is_text(series):
            texts = series.dropna().astype(str)
            if texts.str.len().mean() > 40 or series.nunique() > max(200, 0.5 * len(series)):
                reason = "متن آزاد یا دسته‌ی بسیار پرتنوع"
        elif pd.api.types.is_numeric_dtype(series) and series.nunique() == len(series) \
                and pd.api.types.is_integer_dtype(series):
            reason = "عدد یکتا در هر سطر (احتمالاً شناسه)"
        if reason:
            dropped[col] = reason
    return frame.drop(columns=list(dropped)), dropped


# ------------------------------------------------------------------ هدف

def prepare_target(frame, target, task):
    y = frame[target]
    frame = frame[y.notna()].copy()
    y = frame[target]
    if task == "classification":
        if is_text(y):
            lowered = y.astype(str).str.strip().str.lower()
            positives = {"yes", "y", "true", "1", "churn", "churned", "fraud", "default", "failure", "left"}
            if lowered.nunique() == 2 and lowered.isin(positives).any():
                y = lowered.isin(positives).astype(int)
            else:
                y = lowered
        counts = y.value_counts()
        keep = counts[counts >= max(10, 0.005 * len(y))].index
        frame = frame[y.isin(keep)].copy()
        y = y[y.isin(keep)]
        if y.nunique() < 2:
            raise SystemExit("هدف بعد از پاک‌سازی کمتر از دو کلاس دارد")
        if y.nunique() > 20:
            raise SystemExit("هدف بیش از ۲۰ کلاس دارد؛ برای دسته‌بندی مناسب نیست")
    else:
        y = pd.to_numeric(y, errors="coerce")
        frame = frame[y.notna()].copy()
        y = y[y.notna()]
        if y.nunique() < 10:
            raise SystemExit("هدف عددی تنوع کافی ندارد")
    return frame.drop(columns=[target]), y


def leakage_screen(X, y, task):
    """ویژگی‌ای که به‌تنهایی هدف را تقریباً کامل پیش‌بینی کند، مشکوک به نشت است."""
    from sklearn.metrics import r2_score, roc_auc_score

    leaks = {}
    sample = X.sample(min(len(X), 20_000), random_state=SEED)
    ys = y.loc[sample.index]
    binary = task == "classification" and ys.nunique() == 2
    for col in sample.columns:
        s = sample[col]
        try:
            if is_text(s):
                means = ys.astype(float).groupby(s.astype(str)).transform("mean") if binary or task == "regression" else None
                if means is None:
                    continue
                score_input = means
            else:
                score_input = s.fillna(s.median())
            if binary:
                auc = roc_auc_score(ys, score_input)
                strength = max(auc, 1 - auc)
            elif task == "regression":
                strength = abs(np.corrcoef(score_input.astype(float), ys.astype(float))[0, 1])
            else:
                continue
            if strength >= 0.98:
                leaks[col] = round(float(strength), 3)
        except Exception:
            continue

    # نشت فرمولی: هدف = نسبت یا حاصل‌ضرب دو ستون دیگر. مثال واقعی: در داده‌ی
    # محصول کشاورزی Yield = Production / Area بود و مدل با R² ۰٫۹۸ «موفق» شد —
    # چون جواب را از روی خودش می‌خواند. صورت کسر (نتیجه‌ی رخداد) کنار می‌رود.
    if task == "regression":
        numeric = [c for c in sample.columns
                   if not is_text(sample[c]) and c not in leaks][:20]
        target = ys.astype(float)
        for a in numeric:
            for b in numeric:
                if a == b:
                    continue
                with np.errstate(all="ignore"):
                    va, vb = sample[a].astype(float), sample[b].astype(float)
                    for kind, value in (("ratio", va / vb.replace(0, np.nan)),
                                        ("product", va * vb)):
                        ok = value.notna() & np.isfinite(value) & target.notna()
                        if ok.sum() < 50:
                            continue
                        r = abs(np.corrcoef(value[ok], target[ok])[0, 1])
                        if r >= 0.98:
                            leaks[a] = f"{kind}:{a}{'/' if kind == 'ratio' else '*'}{b}={round(float(r), 3)}"
                            if kind == "product":
                                leaks[b] = leaks[a]
                if a in leaks:
                    break
    return leaks


# ------------------------------------------------------------------ مدل

def build_preprocessor(X):
    """پیش‌پردازش با feature-engine + scikit-learn.

    Winsorizer دُم‌های پرت را می‌چیند (یک رکورد عجیب، مدل خطی را کج می‌کند) و
    RareLabelEncoder دسته‌های کم‌تکرار را در «نادر» جمع می‌کند — وگرنه هر دسته‌ی
    یکتا یک ستون یک‌بارمصرف می‌شود.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    for c in categorical:
        X[c] = X[c].astype(str)

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    categorical_steps = [("impute", SimpleImputer(strategy="most_frequent"))]
    try:
        from feature_engine.encoding import RareLabelEncoder
        from feature_engine.outliers import Winsorizer
        numeric_steps.insert(0, ("winsor", Winsorizer(capping_method="quantiles",
                                                      tail="both", fold=0.01,
                                                      missing_values="ignore")))
        if categorical:
            categorical_steps.append(("rare", RareLabelEncoder(
                tol=0.02, n_categories=1, missing_values="ignore")))
        USED.add("feature-engine")
    except ImportError:
        pass
    numeric_steps.append(("scale", StandardScaler()))
    categorical_steps.append(("onehot", OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)))

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric),
        ("cat", Pipeline(categorical_steps), categorical),
    ]), numeric, categorical


def anomaly_share(X, numeric):
    """سهم رکوردهای ناهنجار با pyOD (ECOD؛ بدون پارامتر و سریع).

    برای گزارش است نه حذف: «۹٪ رکوردها الگوی غیرعادی دارند» به مدیر می‌گوید
    داده‌اش چقدر تمیز است، و اگر خودِ مسئله کشف تقلب باشد، همین نقطه‌ی شروع است.
    """
    if len(numeric) < 2:
        return None
    try:
        from pyod.models.ecod import ECOD
    except ImportError:
        return None
    frame = X[numeric].astype(float)
    frame = frame.fillna(frame.median()).replace([np.inf, -np.inf], 0)
    sample = frame.sample(min(len(frame), 20_000), random_state=SEED)
    try:
        model = ECOD()
        model.fit(sample)
        USED.add("pyOD")
        return round(float(np.mean(model.labels_)), 4)
    except Exception:
        return None


def drift_report(train, test):
    """رانش داده بین آموزش و آزمون با Evidently.

    اگر داده‌ی آزمون (که جدیدتر است) با آموزش فرق اساسی داشته باشد، مدل در
    عمل زودتر از آنچه فکر می‌کنی کهنه می‌شود — و این را باید به مشتری گفت.
    """
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return None
    try:
        columns = list(train.columns)[:40]
        snapshot = Report([DataDriftPreset()]).run(
            current_data=test[columns].sample(min(len(test), 5000), random_state=SEED),
            reference_data=train[columns].sample(min(len(train), 5000), random_state=SEED))
        value = (snapshot.dict().get("metrics") or [{}])[0].get("value") or {}
        USED.add("Evidently")
        return {"drifted_columns": int(value.get("count", 0)),
                "share": round(float(value.get("share", 0)), 3)}
    except Exception:
        return None


def shap_importance(model, X_sample, out_dir):
    """اهمیت ویژگی با SHAP برای مدل‌های درختی + نمودار beeswarm."""
    try:
        import shap
    except ImportError:
        return None, None
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "estimators_") and not hasattr(estimator, "_predictors"):
        return None, None
    try:
        matrix = model.named_steps["prep"].transform(X_sample)
        names = list(model.named_steps["prep"].get_feature_names_out())
        explainer = shap.TreeExplainer(estimator)
        values = np.array(explainer.shap_values(matrix, check_additivity=False))
        if values.ndim == 3:                       # دسته‌بندی: (نمونه، ویژگی، کلاس)
            values = values[:, :, -1]
        order = np.argsort(-np.abs(values).mean(axis=0))[:12]
        top = [{"feature": _clean_name(names[i]),
                "shap": round(float(np.abs(values[:, i]).mean()), 5)} for i in order]

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.facecolor": "#ece7dd", "axes.facecolor": "#ece7dd",
                             "savefig.facecolor": "#ece7dd"})
        plt.figure()
        shap.summary_plot(values, matrix, feature_names=[_clean_name(n) for n in names],
                          max_display=10, show=False, plot_size=(7.5, 5))
        path = os.path.join(out_dir, "figures", "shap.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close("all")
        USED.add("SHAP")
        return top, os.path.relpath(path, out_dir).replace("\\", "/")
    except Exception as exc:
        print("shap skipped:", exc)
        return None, None


def _clean_name(name):
    return re.sub(r"^(num|cat)__", "", str(name))


def candidates(task, n_classes):
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.linear_model import LogisticRegression, Ridge

    if task == "classification":
        return {
            "baseline": DummyClassifier(strategy="prior"),
            "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                                    class_weight="balanced_subsample", random_state=SEED),
            "gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                                                class_weight="balanced", random_state=SEED),
        }
    return {
        "baseline": DummyRegressor(strategy="median"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                               random_state=SEED),
        "gradient_boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                           random_state=SEED),
    }


def primary_metric(task, n_classes):
    if task == "regression":
        return "neg_mean_absolute_error", "MAE"
    return ("roc_auc", "ROC-AUC") if n_classes == 2 else ("f1_macro", "F1-macro")


def bootstrap_ci(fn, y_true, y_pred, rounds=300):
    rng = np.random.default_rng(SEED)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(rounds):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            scores.append(fn(y_true[idx], y_pred[idx]))
        except ValueError:
            continue
    if not scores:
        return None
    return [round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)]


def evaluate(task, n_classes, model, X_test, y_test, threshold=None):
    from sklearn import metrics as M

    out = {}
    if task == "regression":
        pred = model.predict(X_test)
        out["MAE"] = round(float(M.mean_absolute_error(y_test, pred)), 4)
        out["RMSE"] = round(float(np.sqrt(M.mean_squared_error(y_test, pred))), 4)
        out["R2"] = round(float(M.r2_score(y_test, pred)), 4)
        if (np.asarray(y_test) > 0).all():
            out["MAPE_pct"] = round(float(100 * M.mean_absolute_percentage_error(y_test, pred)), 2)
        out["MAE_CI95"] = bootstrap_ci(M.mean_absolute_error, y_test, pred)
        return out, pred

    if n_classes == 2:
        proba = model.predict_proba(X_test)[:, 1]
        out["ROC-AUC"] = round(float(M.roc_auc_score(y_test, proba)), 4)
        out["PR-AUC"] = round(float(M.average_precision_score(y_test, proba)), 4)
        out["ROC-AUC_CI95"] = bootstrap_ci(M.roc_auc_score, y_test, proba)
        out["positive_rate"] = round(float(np.mean(y_test)), 4)
        if threshold is not None:
            pred = (proba >= threshold).astype(int)
            out["threshold"] = round(float(threshold), 3)
            out["precision"] = round(float(M.precision_score(y_test, pred, zero_division=0)), 4)
            out["recall"] = round(float(M.recall_score(y_test, pred, zero_division=0)), 4)
            out["F1"] = round(float(M.f1_score(y_test, pred, zero_division=0)), 4)
        # پوشش ۲۰٪ پرخطرترین: زبان مدیر، نه زبان آمار
        order = np.argsort(-proba)
        top = order[: max(1, int(0.2 * len(order)))]
        positives = max(1, int(np.sum(y_test)))
        out["top20_capture_pct"] = round(float(100 * np.sum(np.asarray(y_test)[top]) / positives), 1)
        # وقتی موارد مثبت اکثریت‌اند، «درصد پیداشده» سقف پایینی دارد و گمراه می‌کند؛
        # «از ۲۰٪ پرریسک، چند درصد واقعاً مثبت بودند» در برابر میانگین کل، درست‌تر است
        out["top20_precision_pct"] = round(float(100 * np.mean(np.asarray(y_test)[top])), 1)
        return out, proba

    pred = model.predict(X_test)
    out["F1-macro"] = round(float(M.f1_score(y_test, pred, average="macro")), 4)
    out["accuracy"] = round(float(M.accuracy_score(y_test, pred)), 4)
    out["F1-macro_CI95"] = bootstrap_ci(lambda a, b: M.f1_score(a, b, average="macro"), y_test, pred)
    return out, pred


# ------------------------------------------------------------------ نمودار

def charts(out_dir, task, n_classes, results, best, y_test, best_pred, importance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = os.path.join(out_dir, "figures")
    os.makedirs(figs, exist_ok=True)
    ink, accent, muted = "#1c1b19", "#b4532a", "#9a958c"
    paper = "#ece7dd"          # همان کاغذ کارت — نمودار باید روی کارت بنشیند، نه وصله بخورد
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": ink,
                         "ytick.color": ink, "figure.dpi": 160,
                         "figure.facecolor": paper, "axes.facecolor": paper,
                         "savefig.facecolor": paper, "legend.frameon": False})
    paths = {}
    metric = results["primary_metric"]

    names = list(results["models"])
    values = [results["models"][n]["test"].get(metric) for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [accent if n == best else (muted if n == "baseline" else "#5b5750") for n in names]
    ax.barh([n.replace("_", " ") for n in names], values, color=colors)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.3f}", va="center", color=ink)
    ax.set_xlabel(f"{metric} on held-out test data" + (" (lower is better)" if metric == "MAE" else ""))
    ax.invert_yaxis()
    fig.tight_layout()
    paths["models"] = os.path.join(figs, "models.png")
    fig.savefig(paths["models"])
    plt.close(fig)

    if importance:
        top = importance[:10][::-1]
        fig, ax = plt.subplots(figsize=(7, 0.42 * len(top) + 1))
        ax.barh([t["feature"][:38] for t in top], [t["importance"] for t in top], color=accent)
        ax.set_xlabel("Permutation importance (drop in score when shuffled)")
        fig.tight_layout()
        paths["importance"] = os.path.join(figs, "importance.png")
        fig.savefig(paths["importance"])
        plt.close(fig)

    if task == "classification" and n_classes == 2:
        order = np.argsort(-np.asarray(best_pred))
        hits = np.cumsum(np.asarray(y_test)[order]) / max(1, np.sum(y_test))
        share = np.arange(1, len(order) + 1) / len(order)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(share * 100, hits * 100, color=accent, lw=2.2, label="model")
        ax.plot([0, 100], [0, 100], color=muted, ls="--", label="no model")
        ax.set_xlabel("% of cases contacted (highest risk first)")
        ax.set_ylabel("% of real positives reached")
        ax.legend(frameon=False)
        fig.tight_layout()
        paths["gains"] = os.path.join(figs, "gains.png")
        fig.savefig(paths["gains"])
        plt.close(fig)
    elif task == "regression":
        yt = np.asarray(y_test)
        sample = np.random.default_rng(SEED).choice(len(yt), min(3000, len(yt)), replace=False)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(yt[sample], np.asarray(best_pred)[sample], s=8, alpha=0.35, color=accent)
        lo, hi = float(np.min(yt)), float(np.max(yt))
        ax.plot([lo, hi], [lo, hi], color=muted, ls="--")
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
        fig.tight_layout()
        paths["fit"] = os.path.join(figs, "fit.png")
        fig.savefig(paths["fit"])
        plt.close(fig)
    return {k: os.path.relpath(v, out_dir).replace("\\", "/") for k, v in paths.items()}


# ------------------------------------------------- سری زمانی: tsfresh + ADTK

def series_profile(frame, date_col, target, out_dir):
    """پروفایل سری زمانیِ هدف: با tsfresh ویژگی‌های آماری، با ADTK نقطه‌ی پرت.

    چرا این دو: tsfresh کتابخانه‌ی استانداردِ استخراج ویژگی از سری زمانی است و
    ADTK برای کشف ناهنجاری روی سریِ زمان‌دار ساخته شده. pyOD روی جدولِ بدون
    ترتیب کار می‌کند؛ این‌جا ترتیب زمان اهمیت دارد، پس ابزار فرق می‌کند.
    """
    if date_col is None:
        return None
    try:
        s = pd.to_numeric(frame[target], errors="coerce")
    except Exception:
        return None
    s = s.dropna()
    if len(s) < 60:
        return None
    dates = pd.to_datetime(frame.loc[s.index, date_col], errors="coerce")
    ok = dates.notna()
    s, dates = s[ok.values], dates[ok]
    if len(s) < 60:
        return None
    daily = pd.Series(s.to_numpy(dtype=float), index=pd.DatetimeIndex(dates.to_numpy())).sort_index()
    daily = daily.resample("D").mean().interpolate(limit_direction="both")
    if len(daily) < 30:
        return None

    out = {"points": int(len(daily)),
           "from": str(daily.index[0].date()), "to": str(daily.index[-1].date())}
    try:
        from tsfresh.feature_extraction import feature_calculators as fc
        values = daily.to_numpy(dtype=float)
        out["tsfresh"] = {
            "trend_slope": round(float(fc.linear_trend(values, [{"attr": "slope"}])[0][1]), 6),
            "autocorr_lag1": round(float(fc.autocorrelation(values, 1)), 3),
            "autocorr_lag7": round(float(fc.autocorrelation(values, 7)), 3),
            "longest_above_mean": int(fc.longest_strike_above_mean(values)),
            "cid_ce": round(float(fc.cid_ce(values, True)), 4),
        }
        USED.add("tsfresh")
    except Exception as exc:
        print("tsfresh skipped:", exc)
    flags = pd.Series(False, index=daily.index)
    try:
        from adtk.data import validate_series
        from adtk.transformer import DoubleRollingAggregate
        series = validate_series(daily)
        shift = DoubleRollingAggregate(agg="median", window=(7, 1),
                                       diff="l1").transform(series)
        values = shift.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        q1, q3 = np.percentile(finite, [25, 75])
        fence = q3 + 3.0 * (q3 - q1)
        flags = pd.Series(np.isfinite(values) & (values > fence), index=daily.index)
        spikes = daily[flags]
        out["adtk"] = {
            "detector": "ADTK DoubleRollingAggregate(median, 7→1, L1) + IQR fence 3×",
            "anomaly_points": int(flags.sum()),
            "anomaly_share_pct": round(100 * float(flags.mean()), 1),
            "last_spikes": [str(d.date()) for d in list(spikes.index)[-3:]],
        }
        USED.add("ADTK")
    except Exception as exc:
        print("ADTK skipped:", exc)
    figure = _series_figure(daily, flags, target, out_dir)
    if figure:
        out["figure"] = figure
    return out or None


def _series_figure(daily, flags, target, out_dir):
    """نمودار سری زمانی با seaborn — سبکِ آماریِ آماده‌اش برای خطِ روند خواناتر
    از matplotlibِ خام است و کد کمتری می‌خواهد."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        print("seaborn skipped:", exc)
        return None
    figs = os.path.join(out_dir, "figures")
    os.makedirs(figs, exist_ok=True)
    paper, ink, accent = "#ece7dd", "#1c1b19", "#b4532a"
    sns.set_theme(style="ticks", rc={"figure.facecolor": paper, "axes.facecolor": paper,
                                     "savefig.facecolor": paper, "font.size": 11})
    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=160)
    sns.lineplot(x=daily.index, y=daily.to_numpy(dtype=float), ax=ax, color=ink, lw=1.4)
    spikes = daily[flags]
    if len(spikes):
        ax.scatter(spikes.index, spikes.to_numpy(dtype=float), color=accent, s=26, zorder=3,
                   label="ADTK anomalies (%d)" % len(spikes))
        ax.legend(frameon=False)
    ax.set_xlabel("")
    ax.set_ylabel(target[:28])
    sns.despine(ax=ax)
    fig.tight_layout()
    path = os.path.join(figs, "series.png")
    fig.savefig(path)
    plt.close(fig)
    USED.add("seaborn")
    return os.path.relpath(path, out_dir).replace(os.sep, "/")


def interactive_chart(out_dir, results, importance):
    """یک نمودار تعاملی HTML با plotly — تحویل به مشتری باید قابل کاوش باشد
    (هاور روی هر ستون عدد دقیق را نشان می‌دهد)؛ کار PNGِ ثابت نیست."""
    if not importance:
        return None
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        print("plotly skipped:", exc)
        return None
    top = importance[:12][::-1]
    fig = go.Figure(go.Bar(
        x=[t["importance"] for t in top], y=[t["feature"][:40] for t in top],
        orientation="h", marker_color="#b4532a",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(title="Permutation importance — %s" % results.get("best_model"),
                      paper_bgcolor="#ece7dd", plot_bgcolor="#ece7dd",
                      font=dict(color="#1c1b19"), margin=dict(l=10, r=10, t=50, b=10))
    path = os.path.join(out_dir, "importance.html")
    fig.write_html(path, include_plotlyjs="cdn")
    USED.add("plotly")
    return os.path.relpath(path, out_dir).replace(os.sep, "/")


def text_profile(frame, columns):
    """اگر ستون متنی هست، با spaCy واقعاً پردازشش کن: موجودیت و نوع کلمه.
    ادعای «NLP» بدون اجرا ممنوع است؛ اگر مدل زبانی نصب نیست، چیزی برنمی‌گردد."""
    if not columns:
        return None
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except Exception as exc:
        print("spaCy skipped:", exc)
        return None
    col = columns[0]
    sample = frame[col].dropna().astype(str).head(400).tolist()
    ents, nouns, tokens = {}, 0, 0
    for doc in nlp.pipe(sample, batch_size=50):
        for ent in doc.ents:
            ents[ent.label_] = ents.get(ent.label_, 0) + 1
        for tok in doc:
            tokens += 1
            if tok.pos_ in ("NOUN", "PROPN"):
                nouns += 1
    USED.add("spaCy")
    return {"column": col, "docs": len(sample), "tokens": tokens,
            "noun_share_pct": round(100 * nouns / max(1, tokens), 1),
            "entities": dict(sorted(ents.items(), key=lambda kv: -kv[1])[:5])}


def automl_check(X_train, y_train, X_test, y_test, task, scoring, minutes=2):
    """جست‌وجوی خودکار مدل، با بودجه‌ی زمانیِ کوتاه و فقط روی داده‌ی کوچک.

    اول TPOT (همان AutoMLِ تکاملیِ معروف) امتحان می‌شود؛ نسخه‌ی ۰.۱۲.۲ با
    scikit-learn 1.9 ناسازگار است و اگر از کار افتاد، به جست‌وجوی نصف‌کننده‌ی
    خودِ scikit-learn برمی‌گردیم. هرچه اجرا شد، نامش در نتیجه ثبت می‌شود —
    ادعای «AutoML» بدون گفتنِ موتورِ واقعی ممنوع است.
    """
    if len(X_train) > 8000 or X_train.shape[1] > 40:
        return {"skipped": "داده برای بودجه‌ی زمانیِ AutoML بزرگ است"}
    num = X_train.select_dtypes("number").fillna(0)
    if num.shape[1] == 0:
        return {"skipped": "ستون عددی برای AutoML نماند"}
    test_num = X_test.select_dtypes("number").reindex(columns=num.columns).fillna(0)

    try:
        from sklearn.metrics import get_scorer
    except Exception:
        return None

    try:
        from tpot import TPOTClassifier, TPOTRegressor
        cls = TPOTClassifier if task == "classification" else TPOTRegressor
        model = cls(max_time_mins=minutes, random_state=SEED, n_jobs=1, verbosity=0)
        model.fit(num, y_train)
        score = float(get_scorer(scoring)(model, test_num, y_test))
        USED.add("TPOT")
        return {"engine": "TPOT", "budget_minutes": minutes,
                "score": round(abs(score), 4),
                "pipeline": str(getattr(model, "fitted_pipeline_", ""))[:300]}
    except Exception as exc:
        print("TPOT unavailable:", exc)

    try:
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.model_selection import HalvingRandomSearchCV
        if task == "classification":
            from sklearn.ensemble import HistGradientBoostingClassifier as Est
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor as Est
        grid = {"learning_rate": [0.03, 0.06, 0.1, 0.2],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [5, 20, 50],
                "l2_regularization": [0.0, 0.5, 2.0]}
        search = HalvingRandomSearchCV(Est(random_state=SEED), grid, cv=3, n_jobs=1,
                                       random_state=SEED, scoring=scoring,
                                       n_candidates=24, factor=3)
        search.fit(num, y_train)
        score = float(get_scorer(scoring)(search.best_estimator_, test_num, y_test))
        return {"engine": "scikit-learn HalvingRandomSearchCV",
                "candidates": 24, "score": round(abs(score), 4),
                "params": {k: v for k, v in search.best_params_.items()}}
    except Exception as exc:
        print("AutoML fallback failed:", exc)
        return None



# ------------------------------------------------------------------ اجرا

def validate_config(cfg):
    """پیکربندی را با pydantic اعتبارسنجی کن: خطای تایپی در lab_config.json
    نباید بعد از ده دقیقه آموزش مدل لو برود؛ باید ثانیه‌ی اول بگیرد."""
    try:
        from pydantic import BaseModel, Field, ValidationError, field_validator
    except Exception as exc:
        print("pydantic skipped:", exc)
        return cfg

    class LabConfig(BaseModel):
        data: str = Field(min_length=1)
        target: str = Field(min_length=1)
        task: str
        drop: list[str] = []
        time_split: bool = True
        title: str = ""
        source: str = ""

        @field_validator("task")
        @classmethod
        def known_task(cls, value):
            if value not in ("classification", "regression"):
                raise ValueError("task باید classification یا regression باشد")
            return value

    try:
        checked = LabConfig(**{k: v for k, v in cfg.items() if k in LabConfig.model_fields})
    except ValidationError as exc:
        raise SystemExit("پیکربندی نامعتبر است:\n" + str(exc))
    USED.add("Pydantic")
    merged = dict(cfg)
    merged.update(checked.model_dump())
    return merged


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(here, "lab_config.json"))
    parser.add_argument("--target")
    parser.add_argument("--out", default=os.path.join(here, "out"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = json.load(handle)
    if args.target:
        cfg["target"] = args.target
    cfg = validate_config(cfg)
    data_path = cfg["data"] if os.path.isabs(cfg["data"]) else os.path.join(here, cfg["data"])
    started = time.time()

    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                         cross_val_score, train_test_split)
    from sklearn.pipeline import Pipeline

    frame = load(data_path)
    rows_loaded = len(frame)
    target, task = cfg["target"], cfg["task"]
    if target not in frame.columns:
        raise SystemExit(f"ستون هدف «{target}» در داده نیست")
    frame = frame.drop(columns=[c for c in cfg.get("drop", []) if c in frame.columns and c != target])

    dates = parse_dates(frame, target)
    frame, notes = engineer(frame, dates)
    order_col = dates[0] if dates else None
    series_frame = frame[[order_col, target]].copy() if order_col else None
    text_cols = [c for c in frame.columns if c != target and is_text(frame[c])]
    time_split = bool(order_col) and cfg.get("time_split", True)
    if time_split:
        frame = frame.sort_values(order_col).reset_index(drop=True)
    frame = frame.drop(columns=dates)

    X, y = prepare_target(frame, target, task)
    X, dropped = clean(X, None)
    leaks = leakage_screen(X, y, task)
    X = X.drop(columns=list(leaks))
    if X.shape[1] == 0:
        raise SystemExit("بعد از پاک‌سازی هیچ ویژگی قابل‌استفاده‌ای نماند")

    n_classes = int(y.nunique()) if task == "classification" else 0
    if time_split:
        cut = int(0.8 * len(X))
        X_train, X_test, y_train, y_test = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED,
            stratify=y if task == "classification" else None)

    pre, numeric, categorical = build_preprocessor(X_train)
    X_test = X_test.copy()
    for c in categorical:
        X_test[c] = X_test[c].astype(str)

    scoring, metric_name = primary_metric(task, n_classes)
    if task == "classification":
        folds = StratifiedKFold(5, shuffle=not time_split, random_state=SEED if not time_split else None)
    else:
        folds = KFold(5, shuffle=not time_split, random_state=SEED if not time_split else None)
    cv_X, cv_y = X_train, y_train
    if len(X_train) > CV_ROWS:
        cv_X = X_train.sample(CV_ROWS, random_state=SEED)
        cv_y = y_train.loc[cv_X.index]
        if time_split:
            cv_X, cv_y = cv_X.sort_index(), cv_y.sort_index()

    results = {"task": task, "target": target, "primary_metric": metric_name,
               "rows_loaded": rows_loaded, "rows_used": int(len(X)),
               "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
               "split": "time-based (last 20% by date)" if time_split else "random stratified 80/20",
               "features_used": {"numeric": numeric, "categorical": categorical},
               "dropped": dropped, "leakage_suspects": leaks, "notes": notes,
               "classes": {str(k): int(v) for k, v in y.value_counts().items()} if task == "classification" else None,
               "models": {}}

    best, best_cv = None, None
    for name, estimator in candidates(task, n_classes).items():
        pipe = Pipeline([("prep", clone(pre)), ("model", estimator)])
        cv = cross_val_score(pipe, cv_X, cv_y, cv=folds, scoring=scoring, n_jobs=1)
        mean = float(np.mean(cv))
        results["models"][name] = {"cv_mean": round(abs(mean), 4), "cv_std": round(float(np.std(cv)), 4)}
        if name != "baseline" and (best_cv is None or mean > best_cv):
            best, best_cv = name, mean
        print(f"{name:20s} CV {metric_name}={abs(mean):.4f} ±{np.std(cv):.4f}", flush=True)

    fitted = {}
    threshold = None
    for name, estimator in candidates(task, n_classes).items():
        pipe = Pipeline([("prep", clone(pre)), ("model", estimator)])
        if name == best and task == "classification" and n_classes == 2:
            oof = cross_val_predict(pipe, cv_X, cv_y, cv=folds, method="predict_proba")[:, 1]
            grid = np.linspace(0.05, 0.95, 91)
            from sklearn.metrics import f1_score
            threshold = float(grid[int(np.argmax([f1_score(cv_y, oof >= t, zero_division=0) for t in grid]))])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        metrics, pred = evaluate(task, n_classes, pipe, X_test, y_test,
                                 threshold if name == best else None)
        results["models"][name]["test"] = metrics
        if name == best:
            best_pred = pred

    results["best_model"] = best
    base = results["models"]["baseline"]["test"][metric_name]
    top = results["models"][best]["test"][metric_name]
    results["baseline_score"], results["best_score"] = base, top
    if metric_name == "MAE" and base:
        results["improvement_pct"] = round(100 * (base - top) / base, 1)
    elif base:
        results["improvement_pct"] = round(100 * (top - base) / base, 1)

    sample = X_test.sample(min(len(X_test), 5000), random_state=SEED)
    perm = permutation_importance(fitted[best], sample, y_test.loc[sample.index],
                                  scoring=scoring, n_repeats=5, random_state=SEED, n_jobs=1)
    importance = sorted(({"feature": c, "importance": round(float(m), 4)}
                         for c, m in zip(sample.columns, perm.importances_mean)),
                        key=lambda r: -r["importance"])
    results["importance"] = importance[:15]

    os.makedirs(args.out, exist_ok=True)
    # کیفیت داده و تفسیر مدل — با pyOD، Evidently و SHAP
    results["anomaly_share"] = anomaly_share(X, numeric)
    results["drift"] = drift_report(X_train, X_test)
    if series_frame is not None:
        series = series_profile(series_frame, order_col, target, args.out)
        if series:
            results["series"] = series
    text = text_profile(frame, text_cols)
    if text:
        results["text_profile"] = text
    automl = automl_check(X_train, y_train, X_test, y_test, task, scoring)
    if automl:
        results["automl"] = automl
    shap_top, shap_figure = shap_importance(
        fitted[best], X_test.sample(min(len(X_test), 800), random_state=SEED), args.out)
    if shap_top:
        results["shap"] = shap_top
    results["figures"] = charts(args.out, task, n_classes, results, best,
                                y_test, best_pred, importance)
    if shap_figure:
        results["figures"]["shap"] = shap_figure
    if results.get("series", {}).get("figure"):
        results["figures"]["series"] = results["series"]["figure"]
    html = interactive_chart(args.out, results, importance)
    if html:
        results["interactive_chart"] = html
    try:
        import joblib
        joblib.dump(fitted[best], os.path.join(args.out, "model.joblib"))
        results["model_file"] = "model.joblib"
    except Exception as exc:
        results["model_file"] = None
        print("model not saved:", exc)

    # شفافیت فنی: چه کتابخانه‌هایی واقعاً اجرا شدند، با نسخه‌شان
    versions = {}
    for name, module in (("pandas", "pandas"), ("scikit-learn", "sklearn"),
                         ("matplotlib", "matplotlib"), ("SHAP", "shap"),
                         ("pyOD", "pyod"), ("feature-engine", "feature_engine"),
                         ("Evidently", "evidently"), ("tsfresh", "tsfresh"),
                         ("ADTK", "adtk"), ("seaborn", "seaborn"),
                         ("plotly", "plotly"), ("spaCy", "spacy"),
                         ("Pydantic", "pydantic"), ("TPOT", "tpot")):
        if name in USED:
            try:
                versions[name] = __import__(module).__version__
            except Exception:
                versions[name] = "?"
    results["stack"] = {
        "libraries": versions,
        "model": best,
        "ml_area": ("classification" if task == "classification" else "regression"),
        "python": sys.version.split()[0],
    }
    results["seconds"] = round(time.time() - started, 1)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"best={best} {metric_name}: baseline {base} -> {top}")


if __name__ == "__main__":
    sys.exit(main())
