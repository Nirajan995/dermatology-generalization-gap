"""Evaluation primitives: grouped linear probes, bootstrap CIs, kNN purity.

These functions are deliberately small and side-effect free so they can be
unit-tested without any of the image data (see tests/test_analysis.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import KNN_K, N_BOOT, SEED, TEST_SIZE

# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def grouped_split(n: int, groups, test_size: float = TEST_SIZE, seed: int = SEED):
    """Single grouped train/test split. Returns (train_idx, test_idx).

    Grouping prevents images from the same lesion / case landing on both sides.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(np.arange(n), groups=np.asarray(groups).astype(str)))


def filter_rare_classes(labels, min_count: int):
    """Boolean mask keeping only classes with >= min_count examples."""
    labels = pd.Series(labels)
    keep = labels.value_counts()
    keep = keep[keep >= min_count].index
    return labels.isin(keep).values


# ---------------------------------------------------------------------------
# Linear probing
# ---------------------------------------------------------------------------

def fit_linear_probe(emb, y, groups, standardize: bool = True,
                     test_size: float = TEST_SIZE, seed: int = SEED):
    """L2-regularised logistic probe on frozen features, grouped split.

    Returns ``(test_idx, y_test, y_pred)``.
    """
    emb = np.asarray(emb)
    y = np.asarray(y)
    tr, te = grouped_split(len(emb), groups, test_size, seed)

    if standardize:
        scaler = StandardScaler().fit(emb[tr])
        x_tr, x_te = scaler.transform(emb[tr]), scaler.transform(emb[te])
    else:
        x_tr, x_te = emb[tr], emb[te]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_tr, y[tr])
    return te, y[te], clf.predict(x_te)


def fit_fewshot_probe(emb, y, groups, k: int = 10, n_draws: int = 5,
                      standardize: bool = True, test_size: float = TEST_SIZE,
                      seed: int = SEED):
    """Few-shot probe: k class-balanced examples per class, averaged over draws.

    Uses the same held-out test split as ``fit_linear_probe`` so the numbers are
    directly comparable; only the *training* pool is subsampled.
    """
    emb = np.asarray(emb)
    y = np.asarray(y)
    tr, te = grouped_split(len(emb), groups, test_size, seed)

    if standardize:
        scaler = StandardScaler().fit(emb[tr])
        x_tr, x_te = scaler.transform(emb[tr]), scaler.transform(emb[te])
    else:
        x_tr, x_te = emb[tr], emb[te]
    y_tr, y_te = y[tr], y[te]

    scores = []
    for draw in range(n_draws):
        rng = np.random.default_rng(draw)
        idx = []
        for cls in np.unique(y_tr):
            pool = np.where(y_tr == cls)[0]
            idx.extend(rng.choice(pool, min(k, len(pool)), replace=False))
        idx = np.asarray(idx)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(x_tr[idx], y_tr[idx])
        scores.append(balanced_accuracy_score(y_te, clf.predict(x_te)))
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true, y_pred, metric_fn, n_boot: int = N_BOOT, seed: int = SEED):
    """Percentile bootstrap over test-set rows.

    Returns ``(point_estimate, boot_mean, lo, hi)``.

    The point estimate is reported alongside the bootstrap mean because for
    balanced accuracy with rare classes the two differ noticeably: resampling
    sometimes drops a small class entirely, which changes the per-class average.
    Report the point estimate; use the percentiles for the interval.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    point = float(metric_fn(y_true, y_pred))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            stats.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:  # metric undefined on a degenerate resample
            continue
    stats = np.asarray(stats, dtype=float)
    return point, float(stats.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def bootstrap_group_gap(y_true, y_pred, group_labels, min_n: int = 3,
                        n_boot: int = N_BOOT, seed: int = SEED):
    """Max-minus-min accuracy across subgroups, with a percentile interval.

    Returns ``(observed_gap, boot_mean_gap, lo, hi, per_group_accuracy)``.

    CAVEAT (report this): ``max - min`` of noisy per-group estimates is biased
    upward, and the bootstrap mean is biased further still. The interval's lower
    bound is therefore almost never zero even when the true gap is zero, so this
    interval must NOT be read as a test that the gap differs from zero. Because
    the argument here is that the tone gap is *small*, the bias is conservative:
    the true gap is, if anything, smaller than reported.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    group_labels = np.asarray(group_labels)
    rng = np.random.default_rng(seed)

    groups = [g for g in np.unique(group_labels) if (group_labels == g).sum() >= min_n]
    per_group = {
        g: float((y_true[group_labels == g] == y_pred[group_labels == g]).mean())
        for g in groups
    }
    observed = max(per_group.values()) - min(per_group.values()) if len(per_group) >= 2 else np.nan

    gaps = []
    for _ in range(n_boot):
        accs = []
        for g in groups:
            mask = group_labels == g
            idx = rng.integers(0, mask.sum(), mask.sum())
            accs.append((y_true[mask][idx] == y_pred[mask][idx]).mean())
        if len(accs) >= 2:
            gaps.append(max(accs) - min(accs))
    gaps = np.asarray(gaps, dtype=float)
    return (float(observed), float(gaps.mean()),
            float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5)),
            per_group)


# ---------------------------------------------------------------------------
# Label-free representation quality
# ---------------------------------------------------------------------------

def chance_purity(labels) -> float:
    """Random-neighbour purity floor, sum_i p_i^2."""
    p = pd.Series(labels).value_counts(normalize=True).values
    return float((p ** 2).sum())


def knn_purity(emb, labels, k: int = KNN_K, n_boot: int = 500, seed: int = SEED):
    """kNN neighbour purity and its lift over the chance floor.

    Returns a dict with ``purity``, ``lift``, ``chance``, ``ci_low``, ``ci_high``.

    Features are standardised and compared with cosine distance. The query point
    itself is excluded from its own neighbourhood.
    """
    emb = np.asarray(emb)
    labels = np.asarray(labels)
    emb_s = StandardScaler().fit_transform(emb)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb_s)
    _, idx = nn.kneighbors(emb_s)
    idx = idx[:, 1:]                       # drop self

    per_point = (labels[idx] == labels[:, None]).mean(axis=1)
    rng = np.random.default_rng(seed)
    boots = [per_point[rng.integers(0, len(per_point), len(per_point))].mean()
             for _ in range(n_boot)]

    chance = chance_purity(labels)
    return {
        "purity": float(per_point.mean()),
        "chance": chance,
        "lift": float(per_point.mean() - chance),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n_classes": len(np.unique(labels)),
        "n_images": len(labels),
    }
