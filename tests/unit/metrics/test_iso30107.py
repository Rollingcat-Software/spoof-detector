"""Unit tests for ISO 30107-3 metrics.

Test cases use synthetic but published-correct numbers so the tests double
as a worked example.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    apcer,
    bpcer,
    acer,
    eer,
    far_at_frr,
    frr_at_far,
    hter,
    roc_curve,
    confusion_matrix,
    classification_report,
)


# Fixture: 10 bona-fide + 10 attacks (5 print, 5 replay)
@pytest.fixture
def fixture_balanced():
    bonafide_scores = [0.95, 0.92, 0.88, 0.91, 0.85, 0.96, 0.87, 0.83, 0.90, 0.93]
    attack_scores = [0.10, 0.15, 0.20, 0.05, 0.30, 0.40, 0.55, 0.45, 0.50, 0.35]
    scores = bonafide_scores + attack_scores
    is_bonafide = [True] * 10 + [False] * 10
    attack_types = [None] * 10 + ["print"] * 5 + ["replay"] * 5
    return np.array(scores), np.array(is_bonafide), attack_types


def test_perfect_separation():
    """If bonafide and attack are perfectly separable, EER == 0."""
    scores = [0.9, 0.8, 0.85, 0.1, 0.2, 0.15]
    is_bonafide = [True, True, True, False, False, False]
    eer_value, _ = eer(scores, is_bonafide)
    assert eer_value == pytest.approx(0.0, abs=0.01)


def test_apcer_at_threshold(fixture_balanced):
    scores, is_bonafide, attack_types = fixture_balanced
    # Threshold 0.5: attacks with score >= 0.5 misclassified as bonafide.
    # attack_scores = [0.10, 0.15, 0.20, 0.05, 0.30] (print) + [0.40, 0.55, 0.45, 0.50, 0.35] (replay)
    # print: 0/5 above 0.5 → APCER_print = 0
    # replay: 2/5 above 0.5 (0.55, 0.50 — note: >= 0.5) → APCER_replay = 0.4
    apcer_max, per_type = apcer(scores, is_bonafide, attack_types, 0.5)
    assert per_type["print"] == pytest.approx(0.0)
    assert per_type["replay"] == pytest.approx(0.4)
    assert apcer_max == pytest.approx(0.4)


def test_bpcer_at_threshold(fixture_balanced):
    scores, is_bonafide, _ = fixture_balanced
    # All bona-fide are >= 0.83, threshold 0.5 → none rejected → BPCER = 0
    assert bpcer(scores, is_bonafide, 0.5) == pytest.approx(0.0)
    # Threshold 0.9 → bona-fide [0.85, 0.87, 0.88, 0.83] reject → 4/10
    assert bpcer(scores, is_bonafide, 0.9) == pytest.approx(0.4)


def test_acer_combines_apcer_bpcer(fixture_balanced):
    scores, is_bonafide, attack_types = fixture_balanced
    acer_v, ap_v, bp_v = acer(scores, is_bonafide, attack_types, 0.5)
    assert acer_v == pytest.approx((ap_v + bp_v) / 2)


def test_eer_lies_on_curve_intersection(fixture_balanced):
    scores, is_bonafide, attack_types = fixture_balanced
    eer_val, eer_th = eer(scores, is_bonafide, attack_types)
    # At EER threshold, APCER ≈ BPCER ≈ EER value
    apcer_at, _ = apcer(scores, is_bonafide, attack_types, eer_th)
    bpcer_at = bpcer(scores, is_bonafide, eer_th)
    assert abs(apcer_at - bpcer_at) < 0.2  # tolerant for small discrete sample
    assert eer_val == pytest.approx((apcer_at + bpcer_at) / 2, abs=0.1)


def test_far_at_frr_returns_consistent_threshold(fixture_balanced):
    scores, is_bonafide, attack_types = fixture_balanced
    far, th = far_at_frr(scores, is_bonafide, attack_types, 0.1)
    bp = bpcer(scores, is_bonafide, th)
    # We asked for 10% FRR — actual BPCER at the threshold should be ~10%
    assert bp == pytest.approx(0.1, abs=0.1)


def test_frr_at_far_returns_consistent_threshold(fixture_balanced):
    scores, is_bonafide, attack_types = fixture_balanced
    frr, th = frr_at_far(scores, is_bonafide, 0.1)
    apcer_at, _ = apcer(scores, is_bonafide, attack_types, th)
    assert apcer_at == pytest.approx(0.1, abs=0.1)


def test_hter_equals_acer_for_binary_fas(fixture_balanced):
    scores, is_bonafide, attack_types = fixture_balanced
    hter_v, _, _ = hter(scores, is_bonafide, attack_types, 0.5)
    acer_v, _, _ = acer(scores, is_bonafide, attack_types, 0.5)
    assert hter_v == pytest.approx(acer_v)


def test_roc_curve_sweeps_thresholds(fixture_balanced):
    scores, is_bonafide, attack_types = fixture_balanced
    roc = roc_curve(scores, is_bonafide, attack_types, n_points=20)
    assert len(roc.points) >= 5
    assert roc.auc > 0.7  # reasonable separation
    assert 0 <= roc.eer <= 1


def test_confusion_matrix_sums_match_dataset(fixture_balanced):
    scores, is_bonafide, attack_types = fixture_balanced
    cm = confusion_matrix(scores, is_bonafide, threshold=0.5, attack_types=attack_types)
    # Row sums should match counts
    assert sum(cm["bonafide"].values()) == 10
    assert sum(cm["attack:print"].values()) == 5
    assert sum(cm["attack:replay"].values()) == 5


def test_classification_report_one_call(fixture_balanced):
    scores, is_bonafide, attack_types = fixture_balanced
    report = classification_report(scores, is_bonafide, attack_types)
    # Spot-check the keys a paper would reference
    for key in ["apcer_max", "bpcer", "acer", "eer", "auc", "n_bonafide", "n_attack"]:
        assert key in report
    assert report["n_bonafide"] == 10
    assert report["n_attack"] == 10


def test_apcer_per_type_max_published_attack_first():
    """Real OULU-NPU style: APCER_max should equal the worst-of-types value."""
    scores = [0.9, 0.91, 0.85,  # 3 bonafide
              0.95, 0.96, 0.92,  # 3 print attacks (very strong — pass through)
              0.05, 0.10, 0.15]  # 3 replay attacks (weak)
    is_bonafide = [True, True, True, False, False, False, False, False, False]
    attack_types = [None, None, None, "print", "print", "print", "replay", "replay", "replay"]
    apcer_max, per_type = apcer(scores, is_bonafide, attack_types, 0.5)
    assert per_type["print"] == pytest.approx(1.0)  # all print attacks fool the system
    assert per_type["replay"] == pytest.approx(0.0)
    assert apcer_max == pytest.approx(1.0)  # APCER_max = worst attack type
