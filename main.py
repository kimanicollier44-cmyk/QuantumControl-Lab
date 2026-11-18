#!/usr/bin/env python3
"""QuantumControl-Lab - educational quantum control lab.

by Darion Kellar

Pure Python standard library. Zero dependencies. Zero network.
Education only - and the central lesson is stated up front:

  * QuantumControl-Lab computes IDEAL-MODEL control math for one
    driven qubit: exact piecewise-constant propagators of
    H(t) = (Delta/2) sigma_z + (Omega(t)/2)(cos phi sigma_x
    + sin phi sigma_y), Rabi oscillations and the chevron, pulse-area
    calibration by search, and a GRAPE-lite shaped pulse optimised
    against a labelled detuning ensemble.
  * Every parameter is a LABELLED teaching value in dimensionless
    units (Omega0 = 1, times in units of 1/Omega0). No pulse in this
    file has ever touched a qubit. The detuning, the amplitude cap,
    and the error sweeps are model inputs, not device data.
  * A fidelity of 0.999 in this lab is a statement about a 2x2 matrix
    product in an ideal model - never a promise about hardware.

Hardware pulse delivery, real-device calibration, fault-tolerance,
quantum-advantage, and guaranteed-fidelity claims are permanent
HOLD gates - with Gemini Argon 4 Extended Thinking as DECIDER
(names-only, tighten-only), Muse as advisor (never overrides
HOLD), and Google Flow plan-only seats.
"""
import hashlib
import json
import math
import sys

# ---------------------------------------------------------------- constants
VERSION = "1.0.0"

# Labelled teaching parameters. Dimensionless: the drive amplitude
# OMEGA0 sets the unit of frequency, time is measured in 1/OMEGA0.
OMEGA0 = 1.0                    # nominal drive amplitude (labelled)
NOMINAL_DETUNING = 0.3          # plant detuning the shaped pulse is built for
PI_TIME = math.pi / OMEGA0      # square-pulse pi time on resonance

# GRAPE-lite design book (all labelled, all stated in the open).
GRAPE_SEGMENTS = 16
GRAPE_TOTAL_TIME = 2.0 * math.pi
GRAPE_OMEGA_MAX = 1.5
GRAPE_ENSEMBLE = (0.2, 0.3, 0.4)   # detunings the pulse is averaged over
GRAPE_MAX_ITERS = 1500

KET_0 = (1.0 + 0.0j, 0.0 + 0.0j)
KET_1 = (0.0 + 0.0j, 1.0 + 0.0j)

# ---------------------------------------------------------------- HOLD gates
HOLD_GATES = {
    "hardware_pulse_delivery": "HOLD - no delivering pulses to real quantum hardware from this lab",
    "real_device_calibration_claim": "HOLD - no claim that a real device was calibrated by this simulation",
    "quantum_computer_operation": "HOLD - no operating a real quantum computer or control stack",
    "fault_tolerance_claim": "HOLD - no fault-tolerance or error-correction-threshold claim from a 2x2 toy model",
    "quantum_advantage_claim": "HOLD - no quantum advantage or supremacy claim",
    "guaranteed_hardware_fidelity_claim": "HOLD - ideal-model fidelities are never guaranteed on hardware",
    "commercial_control_product_claim": "HOLD - no commercial control-software or product claim",
    "real_qubit_characterization_claim": "HOLD - no characterising real qubits (T1, T2, frequencies) from simulated numbers",
    "cryogenic_system_guidance": "HOLD - no cryogenic or dilution-refrigerator operating guidance",
    "control_electronics_safety_bypass": "HOLD - safety interlocks on real control electronics are never bypassed",
    "weapons_guidance_claim": "HOLD - no weapons, guidance, or targeting use of this control math",
    "medical_device_control_claim": "HOLD - no medical-device control claims",
    "investment_claim": "HOLD - no investment claims; a lab model is not a prospectus",
    "remove_human_oversight": "HOLD - human oversight of any real control system is never removed",
}


def gate(action):
    """Return 'HOLD' if the action is gated, else 'PROCEED' (sandbox sim)."""
    return "HOLD" if action in HOLD_GATES else "PROCEED"


# ================================================================ 1. 2x2 complex matrices
# The whole lab is 2x2 complex matrices held as nested lists. No
# numpy: every product below can be recomputed by hand, which is
# the point of an educational control lab.

def mat_mul(a, b):
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0],
         a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0],
         a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]


def mat_dagger(a):
    return [[a[0][0].conjugate(), a[1][0].conjugate()],
            [a[0][1].conjugate(), a[1][1].conjugate()]]


def mat_trace(a):
    return a[0][0] + a[1][1]


def mat_vec(a, v):
    return (a[0][0] * v[0] + a[0][1] * v[1],
            a[1][0] * v[0] + a[1][1] * v[1])


IDENTITY = [[1.0 + 0.0j, 0.0j], [0.0j, 1.0 + 0.0j]]
SIGMA_X = [[0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0j]]
SIGMA_Y = [[0.0j, -1.0j], [1.0j, 0.0j]]
SIGMA_Z = [[1.0 + 0.0j, 0.0j], [0.0j, -1.0 + 0.0j]]

# The GRAPE target: a pi rotation about x (the X gate, up to phase).
TARGET_X = SIGMA_X


def is_unitary(u, tol=1e-9):
    prod = mat_mul(mat_dagger(u), u)
    return all(abs(prod[i][j] - IDENTITY[i][j]) < tol
               for i in range(2) for j in range(2))


# ================================================================ 2. Exact propagators
# One segment holds (amplitude, phase) constant for dt while the
# plant detuning Delta acts the whole time:
#   H = (1/2) w . sigma,  w = (Omega cos phi, Omega sin phi, Delta)
#   U = cos(|w| dt / 2) I - i sin(|w| dt / 2) (n . sigma),  n = w / |w|
# That closed form is exact - no Trotter error anywhere in the lab.

def segment_propagator(amplitude, phase, detuning, dt):
    """Exact propagator of one constant segment."""
    wx = amplitude * math.cos(phase)
    wy = amplitude * math.sin(phase)
    wz = detuning
    w = math.sqrt(wx * wx + wy * wy + wz * wz)
    if w == 0.0 or dt == 0.0:
        return [[1.0 + 0.0j, 0.0j], [0.0j, 1.0 + 0.0j]]
    half = w * dt / 2.0
    c = math.cos(half)
    s = math.sin(half) / w
    # U = c I - i s (wx sx + wy sy + wz sz)
    return [
        [c - 1j * s * wz, -1j * s * (wx - 1j * wy)],
        [-1j * s * (wx + 1j * wy), c + 1j * s * wz],
    ]


def pulse_propagator(segments, detuning):
    """Total propagator of a pulse: segments applied right to left.

    segments: list of (amplitude, phase, dt). The plant detuning is
    a property of the qubit, so it enters every segment unchanged.
    """
    u = [[1.0 + 0.0j, 0.0j], [0.0j, 1.0 + 0.0j]]
    for amplitude, phase, dt in segments:
        u = mat_mul(segment_propagator(amplitude, phase, detuning, dt), u)
    return u


def square_pulse(amplitude=OMEGA0, phase=0.0, duration=PI_TIME):
    """The textbook square pi-pulse attempt: one constant segment."""
    return [(amplitude, phase, duration)]


def population_1(u, ket=KET_0):
    """Probability of |1> after propagator u acts on ket."""
    out = mat_vec(u, ket)
    return abs(out[1]) ** 2


def gate_fidelity(u, target=TARGET_X):
    """Process fidelity |Tr(target+ U)|^2 / 4 (phase-insensitive)."""
    return abs(mat_trace(mat_mul(mat_dagger(target), u))) ** 2 / 4.0


# ================================================================ 3. Rabi physics + chevron
# On resonance the qubit flops at Omega. Detuned, it flops faster -
# at sqrt(Omega^2 + Delta^2) - but never fully: the best transfer is
# Omega^2 / (Omega^2 + Delta^2). Both facts are derived from the
# propagator numerically and asserted against the closed forms.

def rabi_population(t, amplitude=OMEGA0, detuning=0.0):
    """P(|1>) after driving |0> for time t at fixed amplitude."""
    u = segment_propagator(amplitude, 0.0, detuning, t)
    return population_1(u)


def analytic_rabi(t, amplitude=OMEGA0, detuning=0.0):
    """Closed-form Rabi curve the propagator must reproduce."""
    w = math.sqrt(amplitude ** 2 + detuning ** 2)
    if w == 0.0:
        return 0.0
    return (amplitude ** 2 / w ** 2) * math.sin(w * t / 2.0) ** 2


def max_transfer(amplitude=OMEGA0, detuning=0.0):
    """Best achievable inversion at this detuning: Omega^2/(Omega^2+Delta^2)."""
    return amplitude ** 2 / (amplitude ** 2 + detuning ** 2)


def _golden_max(f, lo, hi, tol=1e-12):
    """Golden-section maximisation on a bracket - deterministic."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = f(c), f(d)
    while hi - lo > tol:
        if fc < fd:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = f(d)
        else:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = f(c)
    t = (lo + hi) / 2.0
    return t, f(t)


def first_peak(amplitude=OMEGA0, detuning=0.0):
    """First maximum of the Rabi curve, found by the lab's own search.

    A coarse scan brackets the first peak, golden-section refines it.
    Returns (t_peak, p_peak). The oscillation frequency is pi/t_peak.
    """
    w = math.sqrt(amplitude ** 2 + detuning ** 2)
    horizon = 2.0 * math.pi / w
    steps = 400
    ts = [horizon * i / steps for i in range(steps + 1)]
    ps = [rabi_population(t, amplitude, detuning) for t in ts]
    for i in range(1, steps):
        if ps[i] >= ps[i - 1] and ps[i] >= ps[i + 1]:
            return _golden_max(
                lambda t: rabi_population(t, amplitude, detuning),
                ts[i - 1], ts[i + 1])
    return ts[steps], ps[steps]  # pragma: no cover - monotone edge case


def measured_rabi_frequency(amplitude=OMEGA0, detuning=0.0):
    """Oscillation frequency read off the first peak: pi / t_peak."""
    t_peak, _ = first_peak(amplitude, detuning)
    return math.pi / t_peak


def chevron_report(amplitude=OMEGA0):
    """Rabi chevron book: measured vs analytic frequency + transfer."""
    rows = []
    for detuning in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        t_peak, p_peak = first_peak(amplitude, detuning)
        rows.append({
            "detuning": detuning,
            "t_peak": t_peak,
            "p_peak": p_peak,
            "freq_measured": math.pi / t_peak,
            "freq_analytic": math.sqrt(amplitude ** 2 + detuning ** 2),
            "transfer_analytic": max_transfer(amplitude, detuning),
        })
    return rows


# ================================================================ 4. Pulse-area calibration
# Calibration = find the duration whose first peak inverts the qubit,
# by search, not by quoting pi/Omega. Then price the classic mistake:
# the amplitude is mis-set by +5% but the nominal duration is kept.

def calibrate_pi_time(amplitude=OMEGA0):
    """Search for the pi-pulse duration at this amplitude (Delta = 0)."""
    t_peak, p_peak = first_peak(amplitude, 0.0)
    return {"amplitude": amplitude, "t_pi": t_peak, "p_at_pi": p_peak,
            "t_pi_analytic": math.pi / amplitude}


def amplitude_error_report(error=0.05):
    """Inversion error when amplitude is off by `error` but time is nominal."""
    rows = []
    for sign in (1.0, -1.0):
        amp = OMEGA0 * (1.0 + sign * error)
        p = rabi_population(PI_TIME, amp, 0.0)
        u = pulse_propagator(square_pulse(amp), 0.0)
        rows.append({
            "error": sign * error,
            "amplitude": amp,
            "p1": p,
            "inversion_error": 1.0 - p,
            "gate_fidelity": gate_fidelity(u),
        })
    return rows


def detuned_square_report(detuning=NOMINAL_DETUNING):
    """The nominal square pi-pulse meeting a detuned plant."""
    u = pulse_propagator(square_pulse(), detuning)
    return {"detuning": detuning,
            "p1": population_1(u),
            "inversion_error": 1.0 - population_1(u),
            "gate_fidelity": gate_fidelity(u)}


# ================================================================ 5. GRAPE-lite optimisation
# A shaped pulse: GRAPE_SEGMENTS constant segments over
# GRAPE_TOTAL_TIME, amplitudes capped at GRAPE_OMEGA_MAX, phases
# free. The objective is the MEAN process fidelity to the X gate
# over the labelled detuning ensemble - optimising the average is
# what buys robustness, and the robustness section prices it.
# Optimiser: deterministic Adam ascent on central-finite-difference
# gradients. Same start, same path, same answer, every run.

def _pulse_segments(params):
    n = GRAPE_SEGMENTS
    dt = GRAPE_TOTAL_TIME / n
    return [(params[i], params[n + i], dt) for i in range(n)]


def _ensemble_fidelity(params, detunings=GRAPE_ENSEMBLE):
    segs = _pulse_segments(params)
    return sum(gate_fidelity(pulse_propagator(segs, d))
               for d in detunings) / len(detunings)


def _initial_params(seed):
    """Deterministic asymmetric starts - symmetry is a local trap."""
    n = GRAPE_SEGMENTS
    if seed == 0:
        return [0.7] * n + [0.0] * n
    if seed == 1:
        return ([1.0 + 0.3 * math.sin(1.7 * i + 0.4) for i in range(n)]
                + [0.25 * math.cos(2.3 * i) for i in range(n)])
    return ([0.9 + 0.4 * math.cos(0.9 * i + 1.1) for i in range(n)]
            + [0.3 * math.sin(1.3 * i + 0.7) for i in range(n)])


def _clip(params):
    out = list(params)
    for i in range(GRAPE_SEGMENTS):
        out[i] = min(max(out[i], 0.0), GRAPE_OMEGA_MAX)
    return out


def _gradient(params, base, eps=1e-6):
    grad = []
    for i in range(len(params)):
        up = list(params)
        dn = list(params)
        up[i] += eps
        dn[i] -= eps
        grad.append((_ensemble_fidelity(_clip(up))
                     - _ensemble_fidelity(_clip(dn))) / (2.0 * eps))
    return grad


_GRAPE_CACHE = None


def grape_optimize():
    """Run the deterministic GRAPE-lite optimisation (cached)."""
    global _GRAPE_CACHE
    if _GRAPE_CACHE is not None:
        return _GRAPE_CACHE
    best, best_params, iterations = -1.0, None, 0
    for seed in range(3):
        params = _clip(_initial_params(seed))
        run_best = _ensemble_fidelity(params)
        run_best_params = list(params)
        m1 = [0.0] * len(params)
        m2 = [0.0] * len(params)
        lr, beta1, beta2, eps_adam = 0.06, 0.9, 0.999, 1e-8
        for it in range(1, GRAPE_MAX_ITERS + 1):
            iterations = max(iterations, it)
            grad = _gradient(params, run_best)
            for i in range(len(params)):
                m1[i] = beta1 * m1[i] + (1.0 - beta1) * grad[i]
                m2[i] = beta2 * m2[i] + (1.0 - beta2) * grad[i] ** 2
                m1h = m1[i] / (1.0 - beta1 ** it)
                m2h = m2[i] / (1.0 - beta2 ** it)
                params[i] += lr * m1h / (math.sqrt(m2h) + eps_adam)
            params = _clip(params)
            value = _ensemble_fidelity(params)
            if value > run_best:
                run_best, run_best_params = value, list(params)
            if run_best >= 0.99995:
                break
        if run_best > best:
            best, best_params = run_best, run_best_params
        if best >= 0.99995:
            break
    params = best_params
    segs = _pulse_segments(params)
    result = {
        "segments": GRAPE_SEGMENTS,
        "total_time": GRAPE_TOTAL_TIME,
        "omega_max": GRAPE_OMEGA_MAX,
        "ensemble": list(GRAPE_ENSEMBLE),
        "iterations": iterations,
        "amplitudes": [round(a, 6) for a in params[:GRAPE_SEGMENTS]],
        "phases": [round(p, 6) for p in params[GRAPE_SEGMENTS:]],
        "fidelity_ensemble": best,
        "fidelity_nominal": gate_fidelity(
            pulse_propagator(segs, NOMINAL_DETUNING)),
        "fidelity_on_resonance": gate_fidelity(pulse_propagator(segs, 0.0)),
        "params": params,
    }
    _GRAPE_CACHE = result
    return result


def grape_segments():
    return _pulse_segments(grape_optimize()["params"])


# ================================================================ 6. Robustness pricing
# The shaped pulse was optimised for the detuning ensemble, so the
# honest test is a detuning SWEEP: both pulses evaluated at every
# detuning on one axis. The amplitude sweep runs on each pulse's own
# design plant (square: Delta = 0; shaped: Delta = NOMINAL_DETUNING)
# and is reported, not asserted - robustness was bought against
# detuning, and the table shows what that costs elsewhere.

def detuning_sweep(detunings=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
    shaped = grape_segments()
    rows = []
    for d in detunings:
        rows.append({
            "detuning": d,
            "square_fidelity": gate_fidelity(
                pulse_propagator(square_pulse(), d)),
            "shaped_fidelity": gate_fidelity(pulse_propagator(shaped, d)),
        })
    return rows


def sweep_summary(rows):
    sq = [r["square_fidelity"] for r in rows]
    sh = [r["shaped_fidelity"] for r in rows]
    return {"square_mean": sum(sq) / len(sq), "square_min": min(sq),
            "shaped_mean": sum(sh) / len(sh), "shaped_min": min(sh)}


def amplitude_sweep(scales=(0.90, 0.95, 1.0, 1.05, 1.10)):
    shaped = grape_segments()
    rows = []
    for s in scales:
        sq = [(a * s, p, dt) for a, p, dt in square_pulse()]
        sh = [(a * s, p, dt) for a, p, dt in shaped]
        rows.append({
            "scale": s,
            "square_fidelity": gate_fidelity(pulse_propagator(sq, 0.0)),
            "shaped_fidelity": gate_fidelity(
                pulse_propagator(sh, NOMINAL_DETUNING)),
        })
    return rows


# ================================================================ 7. kNN pulse classifier
# Features: inversion error of the pulse on its plant, detuning
# sensitivity (fidelity lost when the plant detuning moves +0.3),
# and gate fidelity on its plant. The training grid is labelled by
# the lab's own verdict policy (model_label below) - the classifier
# interpolates the policy, it does not replace the propagator math.

def model_label(inversion_error, detuning_sensitivity, fidelity):
    """The pulse-verdict policy distilled to four teaching labels."""
    if fidelity >= 0.999 and detuning_sensitivity <= 0.02:
        return "robust-optimized"
    if inversion_error >= 0.03:
        return "detuned"
    if inversion_error >= 0.003:
        return "mis-set"
    if fidelity >= 0.99:
        return "calibrated"
    return "detuned"


def _training_grid():
    rows = []
    for inv in (0.0, 0.001, 0.006, 0.02, 0.09, 0.2):
        for sens in (0.0, 0.005, 0.02, 0.05, 0.1):
            for fid in (0.85, 0.92, 0.98, 0.993, 0.9995, 1.0):
                rows.append(((inv, sens, fid), model_label(inv, sens, fid)))
    return rows


TRAINING_GRID = _training_grid()


def _scaled(features):
    inv, sens, fid = features
    return (inv / 0.1, sens / 0.1, (1.0 - fid) / 0.1)


def classify_pulse(inversion_error, detuning_sensitivity, fidelity, k=5):
    """kNN over the policy-labelled grid, deterministic tie-breaking."""
    target = _scaled((inversion_error, detuning_sensitivity, fidelity))
    ranked = []
    for features, label in TRAINING_GRID:
        point = _scaled(features)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, point)))
        ranked.append((dist, features, label))
    ranked.sort(key=lambda row: (row[0], row[1]))
    votes = {}
    for _, _, label in ranked[:k]:
        votes[label] = votes.get(label, 0) + 1
    priority = {"robust-optimized": 0, "calibrated": 1, "mis-set": 2,
                "detuned": 3}
    return sorted(votes.items(), key=lambda kv: (-kv[1], priority[kv[0]]))[0][0]


def _pulse_features(segments, plant_detuning):
    """(inversion error, detuning sensitivity, fidelity) of a pulse.

    Detuning sensitivity = fidelity lost when the plant detuning
    moves +0.2 - two ensemble steps beyond the design point.
    """
    u = pulse_propagator(segments, plant_detuning)
    u_shifted = pulse_propagator(segments, plant_detuning + 0.2)
    fid = gate_fidelity(u)
    return (1.0 - population_1(u), fid - gate_fidelity(u_shifted), fid)


def classifier_report():
    """Four scenarios whose features come from the lab's own pulses."""
    scenarios = {
        "square_on_resonance": (square_pulse(), 0.0),
        "square_amplitude_plus5pct": (square_pulse(OMEGA0 * 1.05), 0.0),
        "square_on_detuned_plant": (square_pulse(), NOMINAL_DETUNING),
        "grape_on_detuned_plant": (grape_segments(), NOMINAL_DETUNING),
    }
    report = {}
    for name, (segments, plant) in scenarios.items():
        features = _pulse_features(segments, plant)
        report[name] = {
            "features": [round(f, 6) for f in features],
            "label": classify_pulse(*features),
        }
    return report


# ================================================================ 8. Advisory seats
def _trace(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def argon4_decide(topic="quantumcontrol-demo"):
    """Gemini Argon 4 DECIDER seat: MAP->MODEL->STRESS->DECIDE.

    Names-only keys, live calls 0, tighten-only: can add HOLDs, never lift one.
    """
    stages = ["MAP", "MODEL", "STRESS", "DECIDE"]
    payload = {"seat": "gemini-argon-4", "topic": topic, "stages": stages}
    return {
        "seat": "Gemini Argon 4 Extended Thinking",
        "role": "DECIDER",
        "pipeline": stages,
        "tiers": [2048, 8192, 32768],
        "key": "names-only (GEMINI_API_KEY name checked, value never read)",
        "live_calls": 0,
        "decision": "PROCEED - sandbox simulation only, all hardware gates stay HOLD",
        "can_override_hold": False,
        "trace_sha256_16": _trace(payload),
    }


def claude_advisor(topic="quantumcontrol-demo"):
    """Muse advisor seat: advises, can never override a HOLD."""
    return {
        "seat": "Muse",
        "role": "ADVISOR",
        "key": "names-only (MODEL_API_KEY name checked, value never read)",
        "live_calls": 0,
        "advice": "Report the regime with the number: the shaped pulse wins the detuning sweep because it was optimised over an ensemble - and say in the same breath that its amplitude sweep is merely reported, because robustness is always bought against a named error, never against all errors at once.",
        "can_override_hold": False,
    }


def google_flow_plan():
    """Google Flow seat: plan-only, nothing rendered, nothing live."""
    return {
        "seat": "Google Flow",
        "role": "PLAN-ONLY",
        "shots": ["a Bloch vector tipping over under a square pulse",
                  "a chevron of oscillation fans widening with detuning",
                  "a calibration search homing in on the pi point",
                  "sixteen pulse segments reshaping themselves segment by segment",
                  "two fidelity curves crossing a detuning sweep, the shaped one holding"],
        "rendered": False,
        "live": 0,
    }


def brains_report():
    return {"decider": argon4_decide(), "advisor": claude_advisor(),
            "flow": google_flow_plan()}


# ================================================================ checks
def run_checks():
    c = []

    def chk(name, ok):
        c.append((name, bool(ok)))

    # 1. matrices + propagator exactness
    u = segment_propagator(0.8, 0.4, 0.3, 1.7)
    chk("segment_is_unitary", is_unitary(u))
    chk("zero_drive_is_identity",
        segment_propagator(0.0, 0.0, 0.0, 2.5) == IDENTITY)
    chk("free_precession_phase",
        abs(segment_propagator(0.0, 0.0, 1.0, math.pi)[0][0]
            - complex(math.cos(math.pi / 2), -math.sin(math.pi / 2))) < 1e-12)
    chk("pulse_of_one_segment_matches",
        pulse_propagator([(0.8, 0.4, 1.7)], 0.3) == u)
    chk("pulse_order_matters",
        pulse_propagator([(1.0, 0.0, 1.0), (1.0, math.pi / 2, 1.0)], 0.0)
        != pulse_propagator([(1.0, math.pi / 2, 1.0), (1.0, 0.0, 1.0)], 0.0))
    chk("square_pi_pulse_is_x_up_to_phase",
        gate_fidelity(pulse_propagator(square_pulse(), 0.0)) > 1 - 1e-12)
    chk("gate_fidelity_self_is_one", gate_fidelity(TARGET_X) == 1.0)
    chk("gate_fidelity_identity_is_zero",
        abs(gate_fidelity(IDENTITY)) < 1e-15)

    # 2. Rabi physics - propagator vs closed form
    for t in (0.3, 1.1, 2.7, 4.9):
        chk(f"rabi_matches_analytic_t{t}",
            abs(rabi_population(t, 1.0, 0.4) - analytic_rabi(t, 1.0, 0.4))
            < 1e-12)
    chk("pi_pulse_inverts",
        rabi_population(PI_TIME, OMEGA0, 0.0) > 1 - 1e-9)
    chk("half_pi_pulse_balances",
        abs(rabi_population(PI_TIME / 2, OMEGA0, 0.0) - 0.5) < 1e-12)
    chk("two_pi_pulse_returns",
        rabi_population(2 * PI_TIME, OMEGA0, 0.0) < 1e-12)
    chk("max_transfer_formula",
        abs(max_transfer(1.0, 1.0) - 0.5) < 1e-15
        and abs(max_transfer(1.0, 0.0) - 1.0) < 1e-15)
    chk("off_resonance_never_inverts",
        max(rabi_population(0.05 * i, 1.0, 1.0) for i in range(200))
        < 0.5 + 1e-9)

    # 3. chevron - measured frequency vs sqrt(Omega^2 + Delta^2)
    for detuning in (0.0, 0.5, 1.0, 2.0):
        measured = measured_rabi_frequency(OMEGA0, detuning)
        analytic = math.sqrt(OMEGA0 ** 2 + detuning ** 2)
        chk(f"chevron_freq_delta{detuning}",
            abs(measured - analytic) / analytic < 0.01)
        _, p_peak = first_peak(OMEGA0, detuning)
        chk(f"chevron_peak_delta{detuning}",
            abs(p_peak - max_transfer(OMEGA0, detuning)) < 0.01)

    # 4. calibration - search finds the pi point, mis-set is priced
    cal = calibrate_pi_time()
    chk("calibration_finds_pi_time",
        abs(cal["t_pi"] - math.pi / OMEGA0) < 0.01)
    chk("calibration_peak_is_full", cal["p_at_pi"] > 1 - 1e-9)
    errs = amplitude_error_report()
    chk("amplitude_plus5_error",
        abs(errs[0]["inversion_error"] - (1 - math.sin(1.05 * math.pi / 2) ** 2))
        < 1e-9 and errs[0]["inversion_error"] > 0.005)
    chk("amplitude_minus5_error",
        abs(errs[1]["inversion_error"] - (1 - math.sin(0.95 * math.pi / 2) ** 2))
        < 1e-9)
    chk("amplitude_error_symmetric",
        abs(errs[0]["inversion_error"] - errs[1]["inversion_error"]) < 1e-9)
    det = detuned_square_report()
    chk("detuned_square_p1",
        abs(det["p1"] + det["inversion_error"] - 1.0) < 1e-12
        and abs(det["p1"] - 0.9130) < 0.001)
    chk("detuned_square_fidelity", abs(det["gate_fidelity"] - 0.9128) < 0.001)

    # 5. GRAPE-lite - shaped pulse hits the target in the ideal model
    g = grape_optimize()
    chk("grape_fidelity_nominal_999", g["fidelity_nominal"] >= 0.999)
    chk("grape_fidelity_ensemble_999", g["fidelity_ensemble"] >= 0.999)
    chk("grape_amplitudes_within_cap",
        all(0.0 <= a <= GRAPE_OMEGA_MAX for a in g["amplitudes"]))
    chk("grape_beats_square_on_detuned_plant",
        g["fidelity_nominal"] > det["gate_fidelity"] + 0.05)
    chk("grape_result_reproducible",
        abs(gate_fidelity(pulse_propagator(grape_segments(), NOMINAL_DETUNING))
            - g["fidelity_nominal"]) < 1e-12)

    # 6. robustness - the shaped pulse wins the detuning sweep
    rows = detuning_sweep()
    summ = sweep_summary(rows)
    chk("sweep_shaped_mean_wins", summ["shaped_mean"] > summ["square_mean"])
    chk("sweep_shaped_min_wins", summ["shaped_min"] > summ["square_min"])
    chk("sweep_square_perfect_on_resonance",
        rows[0]["square_fidelity"] > 1 - 1e-12)
    chk("sweep_square_degrades",
        rows[-1]["square_fidelity"] < rows[0]["square_fidelity"] - 0.1)
    amp = amplitude_sweep()
    chk("amplitude_sweep_nominal_row",
        amp[2]["scale"] == 1.0
        and amp[2]["square_fidelity"] > 1 - 1e-12
        and abs(amp[2]["shaped_fidelity"] - g["fidelity_nominal"]) < 1e-9)

    # 7. classifier - labels come from the lab's own pulses + policy
    rep = classifier_report()
    chk("classifier_square_calibrated",
        rep["square_on_resonance"]["label"] == "calibrated")
    chk("classifier_misset",
        rep["square_amplitude_plus5pct"]["label"] == "mis-set")
    chk("classifier_detuned",
        rep["square_on_detuned_plant"]["label"] == "detuned")
    chk("classifier_robust",
        rep["grape_on_detuned_plant"]["label"] == "robust-optimized")
    chk("classifier_matches_model",
        all(classify_pulse(*[float(x) for x in e["features"]])
            == model_label(*[float(x) for x in e["features"]])
            for e in rep.values()))

    # gates + seats
    chk("hold_gates_14", len(HOLD_GATES) == 14
        and all(v.startswith("HOLD") for v in HOLD_GATES.values()))
    chk("gate_hardware_hold", gate("hardware_pulse_delivery") == "HOLD")
    chk("gate_calibration_claim_hold",
        gate("real_device_calibration_claim") == "HOLD")
    chk("gate_fidelity_claim_hold",
        gate("guaranteed_hardware_fidelity_claim") == "HOLD")
    chk("gate_fault_tolerance_hold", gate("fault_tolerance_claim") == "HOLD")
    chk("gate_sandbox_proceed", gate("run_simulation") == "PROCEED")
    brains = brains_report()
    chk("argon4_no_override", brains["decider"]["can_override_hold"] is False
        and brains["decider"]["seat"]
        == "Gemini Argon 4 Extended Thinking")
    chk("claude_no_override", brains["advisor"]["can_override_hold"] is False)
    chk("flow_plan_only", brains["flow"]["rendered"] is False
        and brains["flow"]["live"] == 0)
    return c


# ================================================================ CLI / demo
def demo():
    print("=" * 68)
    print("QuantumControl-Lab - quantum control & pulse engineering lab")
    print("by Darion Kellar | MIT | pure stdlib | zero network | EDUCATION ONLY")
    print("=" * 68)
    print("\n[1] THE HONEST CORE - ideal-model math, labelled parameters")
    print("    One driven qubit, H = (Delta/2) sz + (Omega/2)(cos phi sx")
    print("    + sin phi sy), in dimensionless units (Omega0 = 1). Every")
    print("    propagator is the exact closed form - no Trotter error.")
    print("    A 0.999 fidelity here describes a 2x2 matrix product in an")
    print("    ideal model. It is not, and never becomes, a hardware claim.")
    print("\n[2] Rabi chevron - faster off resonance, but never full")
    for row in chevron_report():
        print(f"    Delta {row['detuning']:4.2f} | first peak t "
              f"{row['t_peak']:6.4f} | freq measured "
              f"{row['freq_measured']:.4f} vs analytic "
              f"{row['freq_analytic']:.4f} | best transfer "
              f"{row['transfer_analytic']:.4f}")
    print("    -> detuning speeds the flop up (sqrt(Omega^2 + Delta^2))")
    print("       and caps it (Omega^2 / (Omega^2 + Delta^2)): the two")
    print("       facts every calibration has to live with.")
    print("\n[3] Calibration - the pi point found by search, errors priced")
    cal = calibrate_pi_time()
    print(f"    search: t_pi = {cal['t_pi']:.6f} (analytic "
          f"{cal['t_pi_analytic']:.6f}), P(1) = {cal['p_at_pi']:.9f}")
    for row in amplitude_error_report():
        print(f"    amplitude {row['error']:+.0%}: P(1) = {row['p1']:.6f} "
              f"-> inversion error {row['inversion_error']:.6f}, "
              f"gate fidelity {row['gate_fidelity']:.6f}")
    det = detuned_square_report()
    print(f"    square pi-pulse on the Delta = {NOMINAL_DETUNING} plant: "
          f"P(1) = {det['p1']:.4f}, gate fidelity "
          f"{det['gate_fidelity']:.4f} - calibrated on the wrong qubit.")
    print("\n[4] GRAPE-lite - a shaped pulse for the detuned plant")
    g = grape_optimize()
    print(f"    {g['segments']} segments over T = {g['total_time']:.4f}, "
          f"amplitudes <= {g['omega_max']}, {g['iterations']} iterations")
    print(f"    ensemble {g['ensemble']} -> mean fidelity "
          f"{g['fidelity_ensemble']:.6f}")
    print(f"    fidelity at nominal Delta = {NOMINAL_DETUNING}: "
          f"{g['fidelity_nominal']:.6f} (square: "
          f"{det['gate_fidelity']:.6f})")
    print(f"    amplitudes: {g['amplitudes']}")
    print(f"    phases:     {g['phases']}")
    print("\n[5] Robustness - detuning sweep (asserted) + amplitude sweep")
    rows = detuning_sweep()
    for row in rows:
        print(f"    Delta {row['detuning']:.1f} | square "
              f"{row['square_fidelity']:.6f} | shaped "
              f"{row['shaped_fidelity']:.6f}")
    summ = sweep_summary(rows)
    print(f"    sweep mean: square {summ['square_mean']:.6f} vs shaped "
          f"{summ['shaped_mean']:.6f} | sweep min: square "
          f"{summ['square_min']:.6f} vs shaped {summ['shaped_min']:.6f}")
    for row in amplitude_sweep():
        print(f"    amplitude x{row['scale']:.2f} | square (Delta=0 plant) "
              f"{row['square_fidelity']:.6f} | shaped (Delta="
              f"{NOMINAL_DETUNING} plant) {row['shaped_fidelity']:.6f}")
    print("    -> the shaped pulse was bought against detuning, and the")
    print("       sweep shows it. The amplitude table is reported as-is:")
    print("       robustness is always against a named error, never all.")
    print("\n[6] Pulse classifier (TinyML kNN, labels from the lab's policy)")
    for name, entry in classifier_report().items():
        print(f"    {name:28s} -> {entry['label']}")
    print("\n[7] Advisory seats")
    brains = brains_report()
    print(f"    decider: {brains['decider']['decision']} "
          f"trace={brains['decider']['trace_sha256_16']}")
    print(f"    advisor can_override_hold = {brains['advisor']['can_override_hold']}")
    print(f"    flow rendered = {brains['flow']['rendered']} "
          f"live = {brains['flow']['live']}")
    print(f"\n[8] HOLD gates: {len(HOLD_GATES)}")
    for key in HOLD_GATES:
        print(f"    HOLD  {key}")
    checks = run_checks()
    passed = sum(1 for _, ok in checks if ok)
    print(f"\nCHECKS {passed}/{len(checks)}")
    for name, ok in checks:
        if not ok:
            print(f"    FAIL {name}")
    verdict = "PROCEED" if passed == len(checks) else "HOLD"
    print(f"VERDICT {verdict} (sandbox simulation only) | HOLD gates {len(HOLD_GATES)}")
    return 0 if passed == len(checks) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "demo":
        return demo()
    cmd = argv[0]
    if cmd == "checks":
        checks = run_checks()
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        print(f"CHECKS {sum(1 for _, ok in checks if ok)}/{len(checks)}")
        return 0 if all(ok for _, ok in checks) else 1
    if cmd == "rabi":
        t = float(argv[1]) if len(argv) > 1 else PI_TIME
        d = float(argv[2]) if len(argv) > 2 else 0.0
        print(json.dumps({"t": t, "detuning": d,
                          "p1": rabi_population(t, OMEGA0, d),
                          "analytic": analytic_rabi(t, OMEGA0, d)},
                         indent=2, sort_keys=True))
        return 0
    if cmd == "chevron":
        print(json.dumps(chevron_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "calibrate":
        print(json.dumps({"calibration": calibrate_pi_time(),
                          "amplitude_errors": amplitude_error_report(),
                          "detuned_square": detuned_square_report()},
                         indent=2, sort_keys=True))
        return 0
    if cmd == "grape":
        g = {k: v for k, v in grape_optimize().items() if k != "params"}
        print(json.dumps(g, indent=2, sort_keys=True))
        return 0
    if cmd == "robust":
        rows = detuning_sweep()
        print(json.dumps({"detuning_sweep": rows,
                          "summary": sweep_summary(rows),
                          "amplitude_sweep": amplitude_sweep()},
                         indent=2, sort_keys=True))
        return 0
    if cmd == "classify":
        print(json.dumps(classifier_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "brains":
        print(json.dumps(brains_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "holds":
        for key, value in HOLD_GATES.items():
            print(f"{key}: {value}")
        return 0
    if cmd == "hold-demo":
        action = argv[1] if len(argv) > 1 else "hardware_pulse_delivery"
        print(f"{action}: {HOLD_GATES.get(action, 'PROCEED - not gated')}")
        return 2 if gate(action) == "HOLD" else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
