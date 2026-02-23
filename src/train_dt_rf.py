import time
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.preprocessing_common import (
    clean_numeric_columns,
    encode_target,
    split_features_target,
    apply_train_iqr_filter,
    log_transform,
)

from src.features_sklearn import one_hot_align

from src.evaluate import (
    compute_basic_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_accuracy_vs_threshold,
    save_metrics,
)


def run(train_path: str, test_path: str) -> list[dict]:
    """
    Trains Decision Tree and Random Forest models,
    evaluates them and returns their metrics for benchmarking.
    """

    # -------------------------
    # Load Data
    # -------------------------
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)

    # -------------------------
    # Preprocessing
    # -------------------------
    train_df = clean_numeric_columns(train_df)
    test_df = clean_numeric_columns(test_df)

    train_df = encode_target(train_df)
    test_df = encode_target(test_df)

    train_X, train_y = split_features_target(train_df)
    test_X, test_y = split_features_target(test_df)

    train_X, train_y, test_X, test_y = apply_train_iqr_filter(
        train_X, train_y, test_X, test_y
    )

    train_X, test_X = log_transform(train_X, test_X)

    # -------------------------
    # One-Hot Encoding
    # -------------------------
    categorical_columns = [
        "Quotedetail.ece_ida",
        "Industry",
        "RadiusofAction",
        "ObjectAbbreviation",
        "ece_conceptid",
    ]

    for col in ["Quotedetail.ece_ida"]:
        if col in train_X.columns:
            train_X[col] = train_X[col].astype(str)
        if col in test_X.columns:
            test_X[col] = test_X[col].astype(str)

    train_X, test_X = one_hot_align(train_X, test_X, categorical_columns)

    results = []

    # =====================================================
    # Decision Tree
    # =====================================================
    dt = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )

    t0 = time.time()
    dt.fit(train_X, train_y)
    train_time_dt = time.time() - t0

    probs_dt = dt.predict_proba(test_X)[:, 1]
    y_pred_dt = (probs_dt >= 0.5).astype(int)

    metrics_dt = compute_basic_metrics(test_y, probs_dt, threshold=0.5)
    metrics_dt["model"] = "DecisionTree"
    metrics_dt["train_seconds"] = float(train_time_dt)
    metrics_dt["n_train"] = int(len(train_X))
    metrics_dt["n_test"] = int(len(test_X))

    print("\n=== Decision Tree ===")
    print(classification_report(test_y, y_pred_dt))

    plot_confusion_matrix(test_y, y_pred_dt, "Confusion Matrix (DT)", "cm_dt.png")
    plot_roc_curve(test_y, probs_dt, "ROC Curve (DT)", "roc_dt.png")
    plot_pr_curve(test_y, probs_dt, "PR Curve (DT)", "pr_dt.png")
    plot_accuracy_vs_threshold(test_y, probs_dt, "Accuracy vs Threshold (DT)", "acc_thr_dt.png")

    save_metrics(metrics_dt, "metrics_dt.json")

    results.append(metrics_dt)

    # =====================================================
    # Random Forest
    # =====================================================
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.time()
    rf.fit(train_X, train_y)
    train_time_rf = time.time() - t0

    probs_rf = rf.predict_proba(test_X)[:, 1]
    y_pred_rf = (probs_rf >= 0.5).astype(int)

    metrics_rf = compute_basic_metrics(test_y, probs_rf, threshold=0.5)
    metrics_rf["model"] = "RandomForest"
    metrics_rf["train_seconds"] = float(train_time_rf)
    metrics_rf["n_train"] = int(len(train_X))
    metrics_rf["n_test"] = int(len(test_X))

    print("\n=== Random Forest ===")
    print(classification_report(test_y, y_pred_rf))

    plot_confusion_matrix(test_y, y_pred_rf, "Confusion Matrix (RF)", "cm_rf.png")
    plot_roc_curve(test_y, probs_rf, "ROC Curve (RF)", "roc_rf.png")
    plot_pr_curve(test_y, probs_rf, "PR Curve (RF)", "pr_rf.png")
    plot_accuracy_vs_threshold(test_y, probs_rf, "Accuracy vs Threshold (RF)", "acc_thr_rf.png")

    save_metrics(metrics_rf, "metrics_rf.json")

    results.append(metrics_rf)

    return results


if __name__ == "__main__":
    train_path = "data/Trainingsdatensatz.xlsx"
    test_path = "data/Testdatensatz.xlsx"

    out = run(train_path, test_path)

    for r in out:
        print(r)