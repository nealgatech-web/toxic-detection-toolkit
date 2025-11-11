from sklearn.metrics import f1_score, precision_score, recall_score

def multilabel_metrics(y_true, y_pred, average="macro"):
    return {
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }
