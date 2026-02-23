import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, RobustScaler

from src.preprocessing_common import (
    clean_numeric_columns,
    encode_target,
    split_features_target,
    apply_train_iqr_filter,
    log_transform,
    COLUMNS_TO_CLEAN,
)

from src.evaluate import (
    compute_basic_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_accuracy_vs_threshold,
    save_metrics,
)


def input_fn(features, labels, training=True, batch_size=100):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)


def run(train_path: str, test_path: str, steps: int = 5000) -> dict:
    # Load
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)

    # Clean
    train_df = clean_numeric_columns(train_df)
    test_df = clean_numeric_columns(test_df)

    # Target
    train_df = encode_target(train_df)
    test_df = encode_target(test_df)

    train_X, train_y = split_features_target(train_df)
    test_X, test_y = split_features_target(test_df)

    # IQR + Log
    train_X, train_y, test_X, test_y = apply_train_iqr_filter(train_X, train_y, test_X, test_y)
    train_X, test_X = log_transform(train_X, test_X)

    # Categorical columns (hash buckets)
    categorical_columns = {
        "Quotedetail.ece_ida": 8192,
        "Industry": 200,
        "RadiusofAction": 10,
        "ObjectAbbreviation": 200,
        "ece_conceptid": 4096,
    }

    # Ensure categorical columns are strings
    for col in categorical_columns.keys():
        if col in train_X.columns:
            train_X[col] = train_X[col].astype(str)
        if col in test_X.columns:
            test_X[col] = test_X[col].astype(str)

    # Numerical columns
    numerical_columns = [
        "Quotedetail.Mietpreis",
        "Quotedetail.SummeNK",
        "Quotedetail.Heizkosten",
        "TermofLeaseYears",
        "CreditRating",
        "TurnoverRent",
    ]

    # Scaling (same logic as your notebook)
    robust_cols = ["CreditRating", "Quotedetail.Heizkosten", "TurnoverRent"]
    robust_scaler = RobustScaler()
    train_X[robust_cols] = robust_scaler.fit_transform(train_X[robust_cols])
    test_X[robust_cols] = robust_scaler.transform(test_X[robust_cols])

    scaler = StandardScaler()
    train_X[["TermofLeaseYears"]] = scaler.fit_transform(train_X[["TermofLeaseYears"]])
    test_X[["TermofLeaseYears"]] = scaler.transform(test_X[["TermofLeaseYears"]])

    # Build TF feature columns
    feature_columns = []

    for name, bucket_size in categorical_columns.items():
        cat_col = tf.feature_column.categorical_column_with_hash_bucket(
            key=name,
            hash_bucket_size=bucket_size,
        )
        emb_col = tf.feature_column.embedding_column(cat_col, dimension=8)
        feature_columns.append(emb_col)

    for name in numerical_columns:
        feature_columns.append(tf.feature_column.numeric_column(name))

    # Model
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[64, 30, 13],
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01),
        n_classes=2,
    )

    logging.getLogger().setLevel(logging.INFO)

    # Train
    t0 = time.time()
    classifier.train(
        input_fn=lambda: input_fn(train_X, train_y, training=True),
        steps=steps,
    )
    train_seconds = time.time() - t0

    # Predict probabilities on test
    pred_dicts = list(classifier.predict(input_fn=lambda: input_fn(test_X, test_y, training=False)))
    probs = np.array([p["probabilities"][1] for p in pred_dicts])
    y_pred = (probs >= 0.5).astype(int)

    # Metrics + plots
    metrics = compute_basic_metrics(test_y, probs, threshold=0.5)
    metrics["model"] = "NeuralNet"
    metrics["train_seconds"] = float(train_seconds)
    metrics["n_train"] = int(len(train_X))
    metrics["n_test"] = int(len(test_X))

    plot_confusion_matrix(test_y, y_pred, "Confusion Matrix (Neural Network)", "cm_nn.png")
    plot_roc_curve(test_y, probs, "ROC Curve (Neural Network)", "roc_nn.png")
    plot_pr_curve(test_y, probs, "PR Curve (Neural Network)", "pr_nn.png")
    plot_accuracy_vs_threshold(test_y, probs, "Accuracy vs Threshold (Neural Network)", "acc_thr_nn.png")

    save_metrics(metrics, "metrics_nn.json")

    return metrics


if __name__ == "__main__":
    train_path = "data/Trainingsdatensatz.xlsx"
    test_path = "data/Testdatensatz.xlsx"
    out = run(train_path, test_path, steps=5000)
    print(out)