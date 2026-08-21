"""Data-free tests for the evaluation primitives.

These run in seconds on synthetic embeddings and need none of the image
datasets, so they can gate every commit:

    pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from dermgap.analysis import (
    bootstrap_ci,
    bootstrap_group_gap,
    chance_purity,
    filter_rare_classes,
    fit_fewshot_probe,
    fit_linear_probe,
    grouped_split,
    knn_purity,
)
from dermgap.taxonomy import SCIN_CATEGORIES, map_scin_category

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def separable_data():
    """Three well-separated Gaussian blobs, 30 groups, 2 images per group."""
    rng = np.random.default_rng(0)
    n_per_class, dim = 40, 16
    centers = rng.normal(0, 6, size=(3, dim))
    emb, y, groups = [], [], []
    gid = 0
    for cls in range(3):
        for i in range(n_per_class):
            if i % 2 == 0:
                gid += 1
            emb.append(centers[cls] + rng.normal(0, 1, dim))
            y.append(f"class{cls}")
            groups.append(f"g{gid}")
    return np.asarray(emb), np.asarray(y), np.asarray(groups)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def test_grouped_split_has_no_group_leakage(separable_data):
    _, _, groups = separable_data
    tr, te = grouped_split(len(groups), groups)
    assert not (set(groups[tr]) & set(groups[te]))
    assert len(tr) + len(te) == len(groups)


def test_grouped_split_is_deterministic(separable_data):
    _, _, groups = separable_data
    a = grouped_split(len(groups), groups)
    b = grouped_split(len(groups), groups)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_filter_rare_classes():
    labels = np.array(["a"] * 10 + ["b"] * 3 + ["c"] * 1)
    mask = filter_rare_classes(labels, min_count=5)
    assert mask.sum() == 10
    assert set(labels[mask]) == {"a"}


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def test_linear_probe_recovers_separable_classes(separable_data):
    emb, y, groups = separable_data
    _, y_te, pred = fit_linear_probe(emb, y, groups)
    assert balanced_accuracy_score(y_te, pred) > 0.9


def test_linear_probe_is_near_chance_on_noise():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(120, 16))
    y = np.array(["a", "b"] * 60)
    groups = np.arange(120).astype(str)
    _, y_te, pred = fit_linear_probe(emb, y, groups)
    assert balanced_accuracy_score(y_te, pred) < 0.75


def test_fewshot_probe_runs_and_is_bounded(separable_data):
    emb, y, groups = separable_data
    mean, std = fit_fewshot_probe(emb, y, groups, k=5, n_draws=3)
    assert 0.0 <= mean <= 1.0
    assert std >= 0.0


def test_standardization_flag_changes_nothing_structurally(separable_data):
    emb, y, groups = separable_data
    _, y_a, _ = fit_linear_probe(emb, y, groups, standardize=True)
    _, y_b, _ = fit_linear_probe(emb, y, groups, standardize=False)
    # same held-out rows either way, so raw vs standardized is comparable
    assert np.array_equal(y_a, y_b)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_ci_brackets_point_estimate():
    y_true = np.array(["a"] * 50 + ["b"] * 50)
    y_pred = np.array(["a"] * 45 + ["b"] * 5 + ["b"] * 48 + ["a"] * 2)
    point, mean, lo, hi = bootstrap_ci(y_true, y_pred, balanced_accuracy_score, n_boot=300)
    assert lo <= point <= hi
    assert abs(point - mean) < 0.05


def test_bootstrap_group_gap_is_positive_even_with_no_true_gap():
    """Documents the upward bias of max-minus-min; guards the caveat in the paper."""
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, 240)
    y_pred = y_true.copy()
    flip = rng.random(240) < 0.25          # identical 25% error rate in every group
    y_pred[flip] = 1 - y_pred[flip]
    tone = np.array(["I-II", "III-IV", "V-VI"] * 80)
    observed, boot_mean, lo, _hi, per_group = bootstrap_group_gap(
        y_true, y_pred, tone, n_boot=400)
    assert len(per_group) == 3
    assert boot_mean >= observed - 1e-9    # bootstrap mean is the more inflated one
    assert lo > 0                          # never zero, by construction


# ---------------------------------------------------------------------------
# kNN purity
# ---------------------------------------------------------------------------

def test_chance_purity_matches_closed_form():
    labels = ["a"] * 75 + ["b"] * 25
    assert chance_purity(labels) == pytest.approx(0.75 ** 2 + 0.25 ** 2)


def test_knn_purity_high_for_separable_low_for_noise(separable_data):
    emb, y, _ = separable_data
    good = knn_purity(emb, y, k=5, n_boot=100)
    assert good["lift"] > 0.5
    assert good["ci_low"] <= good["purity"] <= good["ci_high"]

    rng = np.random.default_rng(4)
    noise = rng.normal(size=emb.shape)
    bad = knn_purity(noise, y, k=5, n_boot=100)
    assert abs(bad["lift"]) < 0.15


def test_knn_purity_excludes_self():
    """With k=1 and duplicated points, purity must reflect the neighbour, not self."""
    emb = np.array([[0.0, 0.0], [0.0, 0.01], [5.0, 5.0], [5.0, 5.01]])
    labels = np.array(["a", "b", "a", "b"])
    res = knn_purity(emb, labels, k=1, n_boot=50)
    assert res["purity"] == pytest.approx(0.0)   # each point's neighbour is the other class


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("Eczema", "inflammatory"),
    ("Insect Bite", "inflammatory"),          # reactive, not infectious
    ("Tinea", "infectious"),
    ("Basal Cell Carcinoma", "neoplastic"),
    ("Purpura", "vascular_purpuric"),
    ("Post-Inflammatory hyperpigmentation", "pigmentary"),
    ("Abscess", "traumatic_other"),
    ("Geographic tongue", "other"),
])
def test_scin_category_mapping(label, expected):
    assert map_scin_category(label) == expected


def test_fallback_keywords_resolve_long_tail():
    assert map_scin_category("Fungal dermatosis") == "infectious"
    assert map_scin_category("Acute-on-chronic dyshidrotic eczema of hands") == "inflammatory"
    assert map_scin_category("Idiopathic guttate hypomelanosis") == "pigmentary"


def test_every_mapping_lands_in_declared_categories():
    from dermgap.taxonomy import SCIN_CATEGORY_MAP
    for label in SCIN_CATEGORY_MAP:
        assert map_scin_category(label) in SCIN_CATEGORIES
