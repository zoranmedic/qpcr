import numpy as np


def recall(true: list[str], pred: list[str], k: int = 10) -> float:
    return sum(p in true for p in pred[:k]) / len(true)


def reciprocal_rank(true: list[str], pred: list[str], k=10) -> float:
    rank = 0
    for i, p in enumerate(pred[:k]):
        if p in true:
            rank = i + 1
            break
    return 1.0 / rank if rank != 0 else 0


def precision_at_k(true: list[str], pred: list[str], k=10) -> float:
    return sum(p in true for p in pred[:k]) / k


def average_precision(true: list[str], pred: list[str]) -> float:
    tps = [p in true for p in pred]
    sum_over_ks = sum([precision_at_k(true, pred, k + 1) * rel for k, rel in enumerate(tps) if rel])
    return sum_over_ks / len(true)


def ndcg(true: list[str], pred: list[str]) -> float:
    dcg = 0
    for i, c in enumerate(pred):
        if c in true:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum([1.0 / np.log2(i + 2) for i, _ in enumerate(true)])
    return dcg / idcg if idcg else dcg
