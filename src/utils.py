"""Ortak yardımcılar: seed, temporal kenar filtreleme, metrikler.

Bu modül hem offline eğitim hem de (2. fazda) streaming çıkarım tarafından
kullanılacak şekilde bağımsız tutulmuştur.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def set_seed(seed: int = 42) -> None:
    """Tekrarlanabilirlik için tüm RNG'leri sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def edges_up_to(edge_index: torch.Tensor, node_ts: torch.Tensor, max_ts: int) -> torch.Tensor:
    """Yalnızca time_step <= max_ts olan düğümler arasındaki kenarları döndür.

    Elliptic'te kenarlar hep aynı time step içindedir; bu filtre temporal
    hijyeni garanti eder (test dönemi kenarları eğitim/val forward pass'ine
    sızamaz).
    """
    src_ok = node_ts[edge_index[0]] <= max_ts
    dst_ok = node_ts[edge_index[1]] <= max_ts
    return edge_index[:, src_ok & dst_ok]


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Validation üzerinde F1'i maksimize eden karar eşiğini seç.

    Returns: (threshold, f1_at_threshold)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve son noktada threshold üretmez; hizala
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    if len(f1) == 0:
        return 0.5, 0.0
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """AUPRC (birincil), F1/precision/recall (eşikli) — illicit=1 pozitif sınıf."""
    y_pred = (scores >= threshold).astype(int)
    return {
        "auprc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "n_pos": int(y_true.sum()),
        "n_total": int(len(y_true)),
    }
