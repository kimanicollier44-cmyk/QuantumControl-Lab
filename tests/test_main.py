"""Tests for QuantumControl-Lab: propagators, Rabi, GRAPE, gates."""
import math

import pytest

import main as m


# ---- matrices + propagators
def test_segment_unitary():
    u = m.segment_propagator(0.8, 0.4, 0.3, 1.7)
    assert m.is_unitary(u)


def test_zero_drive_identity():
    assert m.segment_propagator(0.0, 0.0, 0.0, 2.5) == m.IDENTITY


def test_pulse_order_matters():
    a = m.pulse_propagator([(1.0, 0.0, 1.0), (1.0, math.pi / 2, 1.0)], 0.0)
    b = m.pulse_propagator([(1.0, math.pi / 2, 1.0), (1.0, 0.0, 1.0)], 0.0)
    assert a != b


def test_gate_fidelity_extremes():
    assert m.gate_fidelity(m.TARGET_X) == 1.0
    assert abs(m.gate_fidelity(m.IDENTITY)) < 1e-15


# ---- Rabi physics
def test_rabi_matches_analytic():
    for t in (0.3, 1.1, 2.7, 4.9):
        assert abs(m.rabi_population(t, 1.0, 0.4)
                   - m.analytic_rabi(t, 1.0, 0.4)) < 1e-12


def test_pi_pulse_inverts():
    assert m.rabi_population(m.PI_TIME, m.OMEGA0, 0.0) > 1 - 1e-9


def test_half_pi_balances():
    assert abs(m.rabi_population(m.PI_TIME / 2, m.OMEGA0, 0.0) - 0.5) < 1e-12


@pytest.mark.parametrize("detuning", [0.0, 0.5, 1.0, 2.0])
def test_chevron_frequency(detuning):
    measured = m.measured_rabi_frequency(m.OMEGA0, detuning)
    analytic = math.sqrt(m.OMEGA0 ** 2 + detuning ** 2)
    assert abs(measured - analytic) / analytic < 0.01


@pytest.mark.parametrize("detuning", [0.0, 0.5, 1.0, 2.0])
def test_chevron_peak_transfer(detuning):
    _, p_peak = m.first_peak(m.OMEGA0, detuning)
    assert abs(p_peak - m.max_transfer(m.OMEGA0, detuning)) < 0.01


def test_off_resonance_never_inverts():
    assert max(m.rabi_population(0.05 * i, 1.0, 1.0)
               for i in range(200)) < 0.5 + 1e-9


# ---- calibration
def test_calibration_finds_pi():
    cal = m.calibrate_pi_time()
    assert abs(cal["t_pi"] - math.pi / m.OMEGA0) < 0.01
    assert cal["p_at_pi"] > 1 - 1e-9


def test_amplitude_error_priced():
    rows = m.amplitude_error_report()
    expected = 1 - math.sin(1.05 * math.pi / 2) ** 2
    assert abs(rows[0]["inversion_error"] - expected) < 1e-9
    assert rows[0]["inversion_error"] > 0.005
    assert abs(rows[0]["inversion_error"]
               - rows[1]["inversion_error"]) < 1e-9


def test_detuned_square():
    det = m.detuned_square_report()
    assert abs(det["p1"] - 0.9130) < 0.001
    assert abs(det["gate_fidelity"] - 0.9130) < 0.001


# ---- GRAPE-lite
def test_grape_fidelity():
    g = m.grape_optimize()
    assert g["fidelity_nominal"] >= 0.999
    assert g["fidelity_ensemble"] >= 0.999


def test_grape_within_cap():
    g = m.grape_optimize()
    assert all(0.0 <= a <= m.GRAPE_OMEGA_MAX for a in g["amplitudes"])
    assert len(g["amplitudes"]) == m.GRAPE_SEGMENTS
    assert len(g["phases"]) == m.GRAPE_SEGMENTS


def test_grape_reproducible_from_params():
    g = m.grape_optimize()
    f = m.gate_fidelity(m.pulse_propagator(m.grape_segments(),
                                           m.NOMINAL_DETUNING))
    assert abs(f - g["fidelity_nominal"]) < 1e-12


# ---- robustness
def test_detuning_sweep_shaped_wins():
    summ = m.sweep_summary(m.detuning_sweep())
    assert summ["shaped_mean"] > summ["square_mean"]
    assert summ["shaped_min"] > summ["square_min"]


def test_sweep_square_anchors():
    rows = m.detuning_sweep()
    assert rows[0]["square_fidelity"] > 1 - 1e-12
    assert rows[-1]["square_fidelity"] < rows[0]["square_fidelity"] - 0.1


def test_amplitude_sweep_shape():
    rows = m.amplitude_sweep()
    assert [r["scale"] for r in rows] == [0.90, 0.95, 1.0, 1.05, 1.10]
    assert rows[2]["square_fidelity"] > 1 - 1e-12
    assert all(0.0 <= r["shaped_fidelity"] <= 1.0 for r in rows)


# ---- classifier
def test_classifier_scenarios():
    rep = m.classifier_report()
    assert rep["square_on_resonance"]["label"] == "calibrated"
    assert rep["square_amplitude_plus5pct"]["label"] == "mis-set"
    assert rep["square_on_detuned_plant"]["label"] == "detuned"
    assert rep["grape_on_detuned_plant"]["label"] == "robust-optimized"


def test_classifier_matches_model():
    for entry in m.classifier_report().values():
        feats = tuple(float(x) for x in entry["features"])
        assert m.classify_pulse(*feats) == m.model_label(*feats)


# ---- gates + seats
def test_hold_gates_14():
    assert len(m.HOLD_GATES) == 14
    assert all(v.startswith("HOLD") for v in m.HOLD_GATES.values())


def test_gate_outcomes():
    assert m.gate("hardware_pulse_delivery") == "HOLD"
    assert m.gate("guaranteed_hardware_fidelity_claim") == "HOLD"
    assert m.gate("real_device_calibration_claim") == "HOLD"
    assert m.gate("run_simulation") == "PROCEED"


def test_seats_cannot_override():
    brains = m.brains_report()
    assert brains["decider"]["can_override_hold"] is False
    assert brains["decider"]["seat"] == "Gemini Argon 4 Extended Thinking"
    assert brains["advisor"]["can_override_hold"] is False
    assert brains["flow"]["rendered"] is False
    assert brains["flow"]["live"] == 0


def test_checks_all_pass():
    checks = m.run_checks()
    assert all(ok for _, ok in checks), [n for n, ok in checks if not ok]
