import pandas as pd
import numpy as np


def _safe_div(num, den):
    if den == 0:
        return None
    return round(num / den, 4)


def _status(value, metric: str) -> str:
    if value is None:
        return "PASS"
    if metric == "dpd":
        if value > 0.2:
            return "FAIL"
        if value > 0.1:
            return "WARN"
        return "PASS"
    if metric == "eod":
        if value > 0.2:
            return "FAIL"
        if value > 0.1:
            return "WARN"
        return "PASS"
    if metric == "dir":
        if value < 0.8:
            return "FAIL"
        if value < 0.9:
            return "WARN"
        return "PASS"
    return "PASS"


def _worst_status(statuses):
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def compute_group_metrics(df: pd.DataFrame, label_col: str, prediction_col: str, demographic_col: str) -> dict:
    df = df.copy()
    df[demographic_col] = df[demographic_col].astype(str)

    groups = {}
    prediction_rates = {}

    for group_val, group_df in df.groupby(demographic_col):
        y_true = group_df[label_col].astype(int)
        y_pred = group_df[prediction_col].astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        total = len(group_df)

        accuracy = _safe_div(tp + tn, total)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        fpr = _safe_div(fp, fp + tn)
        fnr = _safe_div(fn, fn + tp)
        positive_rate = _safe_div(int((y_true == 1).sum()), total)
        pred_rate = _safe_div(int((y_pred == 1).sum()), total)

        groups[group_val] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "fnr": fnr,
            "size": total,
            "positive_rate": positive_rate,
        }
        prediction_rates[group_val] = pred_rate

    # Cross-group disparity
    valid_pred_rates = [v for v in prediction_rates.values() if v is not None]
    valid_recalls = [groups[g]["recall"] for g in groups if groups[g]["recall"] is not None]

    if len(valid_pred_rates) >= 2:
        dpd = round(max(valid_pred_rates) - min(valid_pred_rates), 4)
        dir_val = _safe_div(min(valid_pred_rates), max(valid_pred_rates))
    else:
        dpd = None
        dir_val = None

    eod = round(max(valid_recalls) - min(valid_recalls), 4) if len(valid_recalls) >= 2 else None

    dpd_status = _status(dpd, "dpd")
    eod_status = _status(eod, "eod")
    dir_status = _status(dir_val, "dir")

    return {
        "column": demographic_col,
        "groups": groups,
        "disparity": {
            "demographic_parity_difference": dpd,
            "equalized_odds_difference": eod,
            "disparate_impact_ratio": dir_val,
            "dpd_status": dpd_status,
            "eod_status": eod_status,
            "dir_status": dir_status,
            "overall_status": _worst_status([dpd_status, eod_status, dir_status]),
        },
    }


def compute_fairness_metrics(df: pd.DataFrame, label_col: str, prediction_col: str, demographic_cols: list) -> dict:
    return {col: compute_group_metrics(df, label_col, prediction_col, col) for col in demographic_cols}
