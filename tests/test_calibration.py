"""Tests for src/models/calibration.py.

All tests that call ``fit()`` use ``monkeypatch.chdir(tmp_path)`` so the
``models/calibration.pkl`` artifact lands in the temporary directory — the
real project ``models/`` directory is never touched.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.contracts import Prediction
from src.models.calibration import (
    CalibrationDataError,
    MIN_CALIBRATION_EVENTS,
    PulseIQCalibrator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TODAY = date(2024, 6, 1)


def _make_prediction(**overrides) -> Prediction:
    """Return a valid ``Prediction`` with sensible defaults.

    Keyword arguments override individual fields before construction.
    """
    base: dict = {
        "geo_id": "Detroit-MI",
        "geo_name": "Detroit, MI",
        "geo_level": "city",
        "run_date": _TODAY,
        "ess_score": 72.0,
        "score_band": "high",
        "delta_7d": 3.5,
        "delta_30d": -1.2,
        "confidence": "high",
        "early_warning": False,
        "missing_sources": [],
        "stale_sources": [],
        "anomaly_flags": [],
        "granularity_warning": False,
        "model_version": "v1.0.0",
        "feature_version": "v1.0.0",
        "calibrated": False,
        "tier1_score": 0.42,
        "tier2_score": 0.18,
        "tier3_score": 0.08,
        "shap_values": {"bls_jobless_claims_delta": 0.35, "fred_delinquency_rate": 0.20},
    }
    base.update(overrides)
    return Prediction(**base)


def _make_paired_data(n: int = 80) -> tuple[list[float], list[float]]:
    """Return n paired (predicted, actual) score samples.

    Uses a near-identity relationship with slight noise so the isotonic
    regression fit is non-trivial (not a degenerate constant output).
    """
    rng = np.random.default_rng(42)
    predicted = np.linspace(10.0, 90.0, n).tolist()
    actual = (np.linspace(10.0, 90.0, n) + rng.normal(0, 3, n)).clip(0, 100).tolist()
    return predicted, actual


# ---------------------------------------------------------------------------
# fit() — minimum events gate
# ---------------------------------------------------------------------------


def test_fit_raises_with_fewer_than_60_events() -> None:
    """fit() raises CalibrationDataError when fewer than 60 samples are supplied."""
    cal = PulseIQCalibrator()
    short = list(range(59))
    with pytest.raises(CalibrationDataError, match="60"):
        cal.fit(short, short)


def test_fit_raises_exact_59_events() -> None:
    """59 is strictly below the threshold and must raise CalibrationDataError."""
    cal = PulseIQCalibrator()
    assert len(list(range(59))) == 59
    with pytest.raises(CalibrationDataError):
        cal.fit(list(range(59)), list(range(59)))


def test_fit_accepts_exactly_60_events(monkeypatch, tmp_path) -> None:
    """Exactly 60 samples is sufficient — no error should be raised."""
    monkeypatch.chdir(tmp_path)
    cal = PulseIQCalibrator()
    xs = list(range(60))
    cal.fit(xs, xs)   # linear y = x is valid isotonic input


def test_fit_raises_on_mismatched_lengths() -> None:
    """fit() raises ValueError when predictions and events have different lengths."""
    cal = PulseIQCalibrator()
    with pytest.raises(ValueError, match="same length"):
        cal.fit(list(range(60)), list(range(61)))


# ---------------------------------------------------------------------------
# calibrate() — fitted path
# ---------------------------------------------------------------------------


def test_calibrate_sets_calibrated_true(monkeypatch, tmp_path) -> None:
    """calibrate() returns a Prediction with calibrated=True after fit()."""
    monkeypatch.chdir(tmp_path)
    cal = PulseIQCalibrator()
    predicted, actual = _make_paired_data()
    cal.fit(predicted, actual)

    result = cal.calibrate(_make_prediction(ess_score=65.0))
    assert result.calibrated is True


def test_calibrate_score_is_clipped_to_0_100(monkeypatch, tmp_path) -> None:
    """Calibrated ess_score is always in [0, 100] regardless of iso extrapolation."""
    monkeypatch.chdir(tmp_path)
    cal = PulseIQCalibrator()
    # Train on data clustered in the middle; scores at extremes may extrapolate
    predicted, actual = _make_paired_data(80)
    cal.fit(predicted, actual)

    for score in [0.0, 0.1, 99.9, 100.0]:
        result = cal.calibrate(_make_prediction(ess_score=score))
        assert 0.0 <= result.ess_score <= 100.0, (
            f"Calibrated score {result.ess_score} is outside [0, 100] for input {score}"
        )


def test_calibrate_updates_score_band(monkeypatch, tmp_path) -> None:
    """score_band reflects the calibrated score, not the raw input score."""
    monkeypatch.chdir(tmp_path)
    cal = PulseIQCalibrator()
    # Force isotonic to map ~74 → ≥75 by training on (74→76) data
    n = 80
    predicted = [74.0] * n
    actual = [76.0] * n
    cal.fit(predicted, actual)

    # Input prediction claims score_band="elevated" (score 72 < 75),
    # but isotonic will push it above 75 → should become "high"
    result = cal.calibrate(_make_prediction(ess_score=74.0, score_band="elevated"))
    # After calibration the score is ≥75, so band must be "high" or "critical"
    assert result.score_band in {"high", "critical"}, (
        f"Expected band to be updated after calibration; got {result.score_band}"
    )


# ---------------------------------------------------------------------------
# calibrate() — unfitted fallback
# ---------------------------------------------------------------------------


def test_calibrate_fallback_unfitted() -> None:
    """A fresh unfitted PulseIQCalibrator returns calibrated=False and confidence=low."""
    cal = PulseIQCalibrator()
    result = cal.calibrate(_make_prediction(ess_score=65.0, confidence="high"))
    assert result.calibrated is False
    assert result.confidence == "low"


def test_calibrate_fallback_preserves_other_fields() -> None:
    """Fields other than calibrated and confidence are unchanged in the fallback."""
    cal = PulseIQCalibrator()
    pred = _make_prediction(ess_score=55.0, geo_id="Chicago-IL")
    result = cal.calibrate(pred)
    assert result.geo_id == "Chicago-IL"
    assert result.ess_score == pytest.approx(55.0)


# ---------------------------------------------------------------------------
# confidence_from_coverage()
# ---------------------------------------------------------------------------


def test_confidence_high() -> None:
    """tier1_coverage >= 0.8 and no stale sources → 'high'."""
    assert PulseIQCalibrator.confidence_from_coverage(70.0, 0.9, []) == "high"


def test_confidence_medium_stale_tier1() -> None:
    """High tier1_coverage but a Tier 1 source stale → demoted to 'medium'."""
    assert (
        PulseIQCalibrator.confidence_from_coverage(70.0, 0.9, ["bls"]) == "medium"
    )


def test_confidence_medium_stale_tier1_fred() -> None:
    """Stale 'fred' source also demotes from 'high' to 'medium'."""
    assert (
        PulseIQCalibrator.confidence_from_coverage(70.0, 0.85, ["fred"]) == "medium"
    )


def test_confidence_medium_low_coverage() -> None:
    """tier1_coverage in [0.5, 0.8) and no critical stale → 'medium'."""
    assert PulseIQCalibrator.confidence_from_coverage(50.0, 0.6, []) == "medium"


def test_confidence_low() -> None:
    """tier1_coverage < 0.5 → 'low'."""
    assert PulseIQCalibrator.confidence_from_coverage(50.0, 0.3, []) == "low"


def test_confidence_stale_non_tier1_does_not_demote() -> None:
    """Stale 'news' or 'openweather' (Tier 3) does not demote from 'high'."""
    result = PulseIQCalibrator.confidence_from_coverage(70.0, 0.9, ["news", "openweather"])
    assert result == "high"


# ---------------------------------------------------------------------------
# calibration_report()
# ---------------------------------------------------------------------------


def test_calibration_report_structure(monkeypatch, tmp_path) -> None:
    """calibration_report() returns a dict with all required keys and 10 buckets."""
    monkeypatch.chdir(tmp_path)
    cal = PulseIQCalibrator()
    predicted, actual = _make_paired_data(80)
    cal.fit(predicted, actual)

    report = cal.calibration_report()

    assert "n_samples" in report
    assert "buckets" in report
    assert "predicted_means" in report
    assert "actual_means" in report
    assert "counts" in report
    assert "input_range" in report
    assert "calibrated_curve" in report

    assert len(report["buckets"]) == 10
    assert len(report["predicted_means"]) == 10
    assert len(report["actual_means"]) == 10
    assert len(report["counts"]) == 10
    assert report["n_samples"] == 80


def test_calibration_report_counts_sum_to_n_samples(monkeypatch, tmp_path) -> None:
    """Sum of per-bucket counts equals the total number of training samples."""
    monkeypatch.chdir(tmp_path)
    cal = PulseIQCalibrator()
    predicted, actual = _make_paired_data(80)
    cal.fit(predicted, actual)

    report = cal.calibration_report()
    assert sum(report["counts"]) == report["n_samples"]


def test_calibration_report_raises_unfitted() -> None:
    """calibration_report() raises CalibrationDataError before fit() is called."""
    cal = PulseIQCalibrator()
    with pytest.raises(CalibrationDataError):
        cal.calibration_report()


# ---------------------------------------------------------------------------
# load() — fallback path
# ---------------------------------------------------------------------------


def test_load_returns_unfitted_when_file_missing(tmp_path) -> None:
    """load() from a nonexistent path returns an unfitted PulseIQCalibrator."""
    path = tmp_path / "models" / "nonexistent.pkl"
    cal = PulseIQCalibrator.load(path=path)
    assert isinstance(cal, PulseIQCalibrator)
    # Unfitted — calibrate() should activate the fallback
    result = cal.calibrate(_make_prediction())
    assert result.calibrated is False


def test_load_returns_fitted_calibrator(monkeypatch, tmp_path) -> None:
    """load() from a valid pickle returns a fitted calibrator."""
    monkeypatch.chdir(tmp_path)
    cal = PulseIQCalibrator()
    predicted, actual = _make_paired_data(80)
    cal.fit(predicted, actual)

    loaded = PulseIQCalibrator.load(path=tmp_path / "models" / "calibration.pkl")
    assert isinstance(loaded, PulseIQCalibrator)
    result = loaded.calibrate(_make_prediction(ess_score=65.0))
    assert result.calibrated is True
