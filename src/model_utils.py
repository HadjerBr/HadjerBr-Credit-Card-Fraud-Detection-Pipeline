# src/training.py

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.logistic_regression_scratch import LogisticRegressionFromScratch


def train_and_evaluate(X_train,
                       y_train,
                       X_test,
                       y_test,
                       learning_rate: float,
                       iterations: int,
                       threshold: float):
    """
    EXACT reproduction of notebook evaluation:
    - Train model
    - Predict with threshold
    - Manual Accuracy, Precision, Recall, F1
    - ROC-AUC identical to notebook
    """

    model = LogisticRegressionFromScratch(
        learning_rate=learning_rate,
        iterations=iterations
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test, threshold=threshold)
    y_prob = model.predict_proba(X_test)

    # Confusion matrix elements
    TP = np.sum((y_test == 1) & (y_pred == 1))
    TN = np.sum((y_test == 0) & (y_pred == 0))
    FP = np.sum((y_test == 0) & (y_pred == 1))
    FN = np.sum((y_test == 1) & (y_pred == 0))

    # Manual metrics (identical to notebook)
    accuracy = (TP + TN) / len(y_test)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN)     if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # ROC-AUC + curve
    auc = roc_auc_score(y_test, y_prob)
    fprs, tprs, _ = roc_curve(y_test, y_prob)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "fprs": fprs,
        "tprs": tprs,
    }

    return model, metrics
