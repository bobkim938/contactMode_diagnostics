"""Reproducible E1 data generation: calibrated A2 and resolved A3.

python e1_generate.py --output my_e1_run
python e1_generate.py --stage calibrate --output calibration_run
python e1_generate.py --stage generate --calibration calibration_run/calibration/thresholds.json --output dataset_run

A2: three quasi-stick >> slide >> quasi-stick cycles, with randomized preload, initial y, and low/high ratios.
A3: three cycles of normal separation/recontact, with randomized preload, initial y, support center, amplitude, frequency, and phase.
A2 quasi-stick control: tangential force varying while remaining in quasi-stick (F_tangential < mu*F_n, within friction cone).
A3 contact control: keep the object in contact while normal force varying
A3 free control: keep the object free without any contact

Each Episodes contains:
    1. trace.npz: numerical arrays recorded at every step
    2. episode.json: Episode params, timing, simulator info, and acceptance checks
    3. events.json: detected contact-mode transition times or intervals

Each row: (s_i, u_i) -> s_i+1:
    read s_t >> compute applied force u_t >> simulate forward >> record s_t, u_t, time, diagnostic >> mj_step >> record s_t+1 and time_next
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import mujoco
import numpy as np

VERSION = "1.0"
Q_TOUCH = 2.75
PROTOCOL = {
    "version": VERSION,
    "calibration": {
        "fit_preloads_N": [8.0, 10.0, 12.0],
        "validation_preloads_N": [9.0, 11.0],
        "subthreshold_ratios": [0.0, 0.25, 0.5, 0.75, 0.90, 0.96],
        "settle_s": 0.3, "ramp_s": 0.25, "hold_s": 0.25,
        "measurement_last_s": 0.10,
        "quasi_ceiling_multiplier": 1.25,
        "slide_floor_multiple_of_quasi": 4.0,
        "minimum_slide_s": 0.02,
        "quasi_sensitivity_multipliers": [1.10, 1.25, 1.50],
        "slide_sensitivity_multiples_of_primary_quasi": [3.0, 4.0, 6.0],
        "rho_primary": 0.98, "rho_sensitivity": [0.97, 0.98, 0.99],
        "loaded_force_N": 0.1, "loaded_force_sensitivity_N": [0.05, 0.1, 0.2],
    },
    "a2": {
        "preload_N": [8.0, 12.0], "initial_y_m": [-0.15, 0.15],
        "settle_s": 0.3, "warmup_ramp_s": 0.2, "warmup_hold_s": 0.2,
        "low_ratio": [0.30, 0.50], "high_ratio": [1.025, 1.040],
        "rise_s": [0.30, 0.40], "high_hold_s": [0.010, 0.020],
        "fall_s": [0.050, 0.080], "low_hold_s": [0.30, 0.50],
        "control_high_ratio": [0.75, 0.94],
        "minimum_slide_dwell_s": 0.020, "minimum_final_quasi_dwell_s": 0.050,
    },
    "a3": {
        "stiffness_N_m": 1000.0, "damping_N_s_m": 30.0,
        "nominal_preload_N": [8.0, 12.0], "initial_y_m": [-0.15, 0.15],
        "amplitude_m": [0.015, 0.025], "frequency_Hz": [1.5, 2.5],
        "settle_s": 0.3, "fade_s": 0.5,
        "discard_periods_after_fade": 1,
        "contact_control_amplitude_m": [0.001, 0.003],
        "free_control_initial_gap_m": 0.08,
        "free_control_amplitude_m": [0.012, 0.020],
        "free_control_frequency_Hz": [4.0, 5.5],
        "minimum_max_gap_m": 0.001, "minimum_free_duration_s": 0.020,
        "minimum_free_samples": 40, "minimum_contact_duration_s": 0.050,
        "recontact_impulse_window_s": 0.020,
        "reference_step_ratio": 0.5,
        "event_time_tolerance_s": 0.001,
        "free_duration_tolerance_s": 0.001,
        "gap_relative_tolerance": 0.02, "gap_absolute_tolerance_m": 0.00005,
        "impulse_relative_tolerance": 0.05, "impulse_absolute_tolerance_Ns": 0.005,
        "same_mode_state_max_error_m": 0.0001,
        "same_mode_velocity_max_error_m_s": 0.01,
        "gap_sensitivity_m": [0.0005, 0.001, 0.002],
        "free_duration_sensitivity_s": [0.010, 0.020, 0.040],
    },
    "startup_rule": {
        "a2": "Exclude all intervals before the fixed settle + warmup ramp + warmup hold, 0.7 s.",
        "a3": "Exclude settle and fade, then one full drive period; start at the next zero-phase ascending cycle boundary.",
        "common": "Keep all rows. analysis_mask selects intervals wholly within the predefined analysis window; rejection excludes the whole episode from accepted splits.",
    },
    "labels": {
        "mode_codes": {"FREE": 0, "QUASI_STICK": 1, "BOUNDARY": 2, "SLIDE": 3, "CONTACT": 4},
        "a2_event_rule": "Report [last confident source sample, first confident destination sample] as an interval, not an exact physical transition time.",
        "a3_event_rule": "Geometric contact-flag entry/exit at sampled states; require resolved gap/duration and refinement agreement.",
    },
}


def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, np.generic):
        return json_safe(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_json(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def smoother(r):
    r = min(1.0, max(0.0, r))
    return r**3 * (10 + r * (-15 + 6 * r))


def uniform(rng, bounds):
    return float(rng.uniform(*bounds))


def build_segments(parameters):
    """Build a continuous, zero-slope-at-endpoints force schedule."""
    segments = []
    cursor = 0.0
    def add(duration, before, after, phase, cycle=-1):
        nonlocal cursor
        segments.append({"start": cursor, "end": cursor + duration,
                         "before": before, "after": after, "phase": phase, "cycle": cycle})
        cursor += duration
    if parameters["kind"] == "calibration_hold":
        c = PROTOCOL["calibration"]
        force = parameters["ratio"] * parameters["mu"] * parameters["preload"]
        add(c["settle_s"], 0, 0, "SETTLE")
        add(c["ramp_s"], 0, force, "RAMP")
        add(c["hold_s"], force, force, "HOLD")
        start = cursor - c["measurement_last_s"]
        windows = []
    else:
        c = PROTOCOL["a2"]
        low = parameters["initial_low_ratio"] * parameters["mu"] * parameters["preload"]
        add(c["settle_s"], 0, 0, "SETTLE")
        add(c["warmup_ramp_s"], 0, low, "WARMUP_RAMP")
        add(c["warmup_hold_s"], low, low, "WARMUP_HOLD")
        start = cursor
        windows = []
        for index, cycle in enumerate(parameters["cycles"]):
            cycle_start = cursor
            high = cycle["high_ratio"] * parameters["mu"] * parameters["preload"]
            new_low = cycle["low_ratio"] * parameters["mu"] * parameters["preload"]
            add(cycle["rise_s"], low, high, "RAMP_UP", index)
            add(cycle["high_hold_s"], high, high, "HIGH_HOLD", index)
            add(cycle["fall_s"], high, new_low, "RAMP_DOWN", index)
            add(cycle["low_hold_s"], new_low, new_low, "LOW_HOLD", index)
            windows.append([cycle_start, cursor])
            low = new_low
    return {"segments": segments, "analysis_start": start,
            "analysis_end": cursor, "duration": cursor, "cycle_windows": windows}


def build_program(parameters):
    if parameters["kind"].startswith("a2") or parameters["kind"] == "calibration_hold":
        return build_segments(parameters)
    c = PROTOCOL["a3"]
    omega = 2 * math.pi * parameters["frequency"]
    period = 1 / parameters["frequency"]
    earliest = c["settle_s"] + c["fade_s"] + c["discard_periods_after_fade"] * period
    winding = math.ceil((omega * (earliest - c["settle_s"]) + parameters["phase"]) / (2 * math.pi) - 1e-12)
    start = c["settle_s"] + (2 * math.pi * winding - parameters["phase"]) / omega
    end = start + parameters["cycle_count"] * period
    return {"analysis_start": start, "analysis_end": end, "duration": end,
            "cycle_windows": [[start + i * period, start + (i + 1) * period]
                              for i in range(parameters["cycle_count"])]}


def support_motion(t, parameters):
    c = PROTOCOL["a3"]
    center = parameters["support_center"]
    tau = t - c["settle_s"]
    if tau <= 0:
        return center, 0.0
    r = min(tau / c["fade_s"], 1.0)
    envelope = smoother(r)
    rate = 30 * r*r * (1-r)**2 / c["fade_s"] if r < 1 else 0.0
    omega = 2 * math.pi * parameters["frequency"]
    angle = omega * tau + parameters["phase"]
    amp = parameters["amplitude"]
    return (center + amp * envelope * math.sin(angle),
            amp * (rate * math.sin(angle) + envelope * omega * math.cos(angle)))


def model_info(model_path, dt):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = dt
    if (model.nq, model.nv, model.nu) != (2, 2, 2):
        raise ValueError("Expected the supplied two-slide-joint, two-force-motor model.")
    for name, index in [("slide_x", 0), ("slide_y", 1)]:
        joint = model.joint(name)
        if int(joint.qposadr[0]) != index or int(joint.dofadr[0]) != index:
            raise ValueError("Joint order differs from the supplied model.")
    if int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_EULER):
        raise ValueError("This protocol is frozen for Euler; recalibrate a revised protocol for another integrator.")
    if model.na != 0 or np.any(model.actuator_ctrllimited) or np.any(model.actuator_forcelimited):
        raise ValueError("Expected stateless motors without control/force clipping, so ctrl is the applied force.")
    if not np.allclose(model.actuator_gainprm[:, 0], 1) or not np.allclose(model.actuator_biasprm, 0):
        raise ValueError("Expected unit-gain unbiased force motors.")
    if not np.allclose(model.jnt_axis, [[1, 0, 0], [0, 1, 0]]):
        raise ValueError("Expected the given world-aligned x/y slide axes.")
    wall = model.geom("wall_geom").id
    sphere = model.geom("object_geom").id
    for index, name in enumerate(["force_x", "force_y"]):
        actuator = model.actuator(name)
        if actuator.id != index or not np.allclose(model.actuator_gear[index], [1, 0, 0, 0, 0, 0]):
            raise ValueError("Expected gear-1 force_x and force_y motors in the supplied order.")
    data = mujoco.MjData(model)
    data.qpos[:] = [Q_TOUCH, 0]
    mujoco.mj_forward(model, data)
    mu = None
    for j in range(data.ncon):
        contact = data.contact[j]
        if {int(contact.geom1), int(contact.geom2)} == {wall, sphere}:
            if not np.isclose(contact.friction[0], contact.friction[1]):
                raise ValueError("This protocol assumes isotropic sliding friction.")
            mu = float(contact.friction[0])
    if mu is None or abs(mujoco.mj_geomDistance(model, data, wall, sphere, 100, None)) > 1e-8:
        raise ValueError("Q_TOUCH=2.75 does not touch the wall in this XML.")
    signature = {"model_sha256": digest_bytes(Path(model_path).read_bytes()),
                 "mujoco_version": mujoco.__version__, "timestep": dt,
                 "protocol_sha256": digest_json(PROTOCOL), "mu": mu}
    return model, wall, sphere, signature


def simulate(model_path, dt, parameters):
    model, wall, sphere, signature = model_info(model_path, dt)
    data = mujoco.MjData(model)
    data.qpos[:] = [parameters.get("initial_qx", Q_TOUCH), parameters.get("initial_y", 0)]
    data.qvel[:] = 0
    program = build_program(parameters)
    count = int(math.ceil((program["duration"] - 1e-12) / dt))
    scalar_fields = ["time", "time_next", "normal_force", "tangential_force", "rho", "gap",
                     "contact_distance", "support_pos", "support_vel", "spring_force", "damping_force"]
    r = {key: np.full(count, np.nan) for key in scalar_fields}
    for key in ["qpos", "qvel", "qacc", "ctrl", "qpos_next", "qvel_next"]:
        r[key] = np.full((count, 2), np.nan)
    r["contact_flag"] = np.zeros(count, dtype=bool)
    r["cycle_id"] = np.full(count, -1, dtype=np.int32)
    r["phase"] = np.full(count, "", dtype="U20")
    r["solver_iterations"] = np.zeros(count, dtype=np.int32)
    buffer = np.zeros(6)
    segment_index = 0
    written = 0
    error = None
    try:
        for i in range(count):
            t = float(data.time)
            support = support_vel = spring = damping = np.nan
            if "segments" in program:
                segments = program["segments"]
                while segment_index + 1 < len(segments) and t >= segments[segment_index]["end"] - 1e-12:
                    segment_index += 1
                seg = segments[segment_index]
                ratio = (t - seg["start"]) / (seg["end"] - seg["start"])
                data.ctrl[:] = [parameters["preload"], seg["before"] + (seg["after"] - seg["before"]) * smoother(ratio)]
                cycle_id, phase = seg["cycle"], seg["phase"]
            else:
                support, support_vel = support_motion(t, parameters)
                spring = PROTOCOL["a3"]["stiffness_N_m"] * (support - data.qpos[0])
                damping = PROTOCOL["a3"]["damping_N_s_m"] * (support_vel - data.qvel[0])
                data.ctrl[:] = [spring + damping, 0]
                cycle_id = min(parameters["cycle_count"] - 1, int(math.floor((t - program["analysis_start"]) * parameters["frequency"] + 1e-10)))
                if cycle_id < 0:
                    cycle_id = -1
                phase = "ANALYSIS" if cycle_id >= 0 else "STARTUP"
            mujoco.mj_forward(model, data)
            fn = ft = capacity = 0.0
            distances = []
            for j in range(data.ncon):
                contact = data.contact[j]
                if {int(contact.geom1), int(contact.geom2)} != {wall, sphere}:
                    continue
                distances.append(float(contact.dist))
                mujoco.mj_contactForce(model, data, j, buffer)
                fn += float(buffer[0])
                ft += float(np.linalg.norm(buffer[1:3]))
                capacity += float(contact.friction[0] * buffer[0])
            values = dict(time=t, qpos=data.qpos, qvel=data.qvel, qacc=data.qacc, ctrl=data.ctrl,
                          normal_force=fn, tangential_force=ft,
                          rho=ft/capacity if capacity > 1e-8 else np.nan,
                          gap=float(mujoco.mj_geomDistance(model, data, wall, sphere, 100, None)),
                          contact_distance=min(distances) if distances else np.nan,
                          contact_flag=bool(distances), cycle_id=cycle_id, phase=phase,
                          support_pos=support, support_vel=support_vel, spring_force=spring, damping_force=damping,
                          solver_iterations=int(np.max(data.solver_niter)))
            for key, value in values.items():
                r[key][i] = value
            written = i + 1
            mujoco.mj_step(model, data)
            r["time_next"][i] = data.time
            r["qpos_next"][i] = data.qpos
            r["qvel_next"][i] = data.qvel
            if not np.all(np.isfinite(np.r_[data.qpos, data.qvel])) or abs(data.time - t - dt) > 1e-9:
                raise RuntimeError("Nonfinite state or unexpected simulator time/reset.")
            if np.any(data.warning.number):
                raise RuntimeError("MuJoCo reported a simulation warning; see warning_counts.")
    except Exception:
        error = traceback.format_exc()
    r = {key: value[:written] for key, value in r.items()}
    valid = np.ones(written, dtype=bool)
    for key in ["qpos", "qvel", "ctrl", "qpos_next", "qvel_next", "qacc"]:
        valid &= np.all(np.isfinite(r[key]), axis=1)
    valid &= np.isfinite(r["time_next"]) & (r["time_next"] > r["time"])
    r["valid_transition"] = valid
    r["startup_mask"] = r["time"] < program["analysis_start"] - 1e-10
    r["analysis_mask"] = valid & ~r["startup_mask"] & (r["time_next"] <= program["analysis_end"] + 1e-10)
    terminal = {}
    if error is None:
        mujoco.mj_forward(model, data)
        terminal = {"time": float(data.time),
                    "gap": float(mujoco.mj_geomDistance(model, data, wall, sphere, 100, None)),
                    "contact_flag": any({int(data.contact[j].geom1), int(data.contact[j].geom2)} == {wall, sphere} for j in range(data.ncon))}
    info = {"parameters": parameters, "program": program, "signature": signature,
            "steps": written, "requested_steps": count, "error": error,
            "terminal": terminal, "warning_counts": data.warning.number.tolist(),
            "solver_iteration_limit": int(model.opt.iterations)}
    return r, info


def label_trace(r, thresholds, kind):
    c = thresholds["primary"]
    mode = np.full(len(r["time"]), 2, dtype=np.int8)
    mode[~r["contact_flag"]] = 0
    loaded = r["contact_flag"] & (r["normal_force"] > c["loaded_force_N"])
    if kind.startswith("a3"):
        mode[loaded] = 4
    else:
        speed = abs(r["qvel"][:, 1])
        mode[loaded & (r["rho"] < c["rho_boundary"]) & (speed <= c["quasi_speed_m_s"])] = 1
        mode[loaded & (r["rho"] >= c["rho_boundary"]) & (speed >= c["slide_speed_m_s"])] = 3
    mode[~r["valid_transition"]] = 2
    r["mode"] = mode


def runs_of(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def basic_reasons(r, info):
    reasons = []
    if info["error"]:
        reasons.append("simulation_error")
    if len(r["time"]) != info["requested_steps"] or not np.all(r["valid_transition"]):
        reasons.append("incomplete_or_invalid_transitions")
    if not np.any(r["analysis_mask"]):
        reasons.append("no_analysis_samples")
    if any(info["warning_counts"]):
        reasons.append("simulator_warning")
    if len(r["time"]) and np.max(r["solver_iterations"]) >= info["solver_iteration_limit"]:
        reasons.append("solver_iteration_limit_reached")
    return reasons


def assess_a2(r, info, thresholds):
    reasons = basic_reasons(r, info)
    mask = r["analysis_mask"]
    c = thresholds["primary"]
    if np.any(mask & ~r["contact_flag"]) or not info["terminal"].get("contact_flag", False):
        reasons.append("contact_loss")
    if np.any(mask & (r["normal_force"] <= c["loaded_force_N"])):
        reasons.append("unloaded_contact")
    dt = info["signature"]["timestep"]
    cycle_results, events = [], []
    for ci, (start, end) in enumerate(info["program"]["cycle_windows"]):
        idx = np.flatnonzero(mask & (r["cycle_id"] == ci))
        codes = r["mode"][idx]
        q = np.flatnonzero(codes == 1)
        s = np.flatnonzero(codes == 3)
        result = {"cycle": ci, "start": start, "end": end, "slide_intervals": len(runs_of(codes == 3))}
        if info["parameters"]["kind"] == "a2_quasi_control":
            if not len(idx) or not np.all(codes == 1):
                reasons.append(f"cycle_{ci}_not_quasi_stick_control")
        else:
            slide_dwell = max((b-a)*dt for a, b in runs_of(codes == 3)) if len(s) else 0.0
            result["longest_slide_dwell_s"] = slide_dwell
            if len(runs_of(codes == 3)) != 1 or slide_dwell < PROTOCOL["a2"]["minimum_slide_dwell_s"]:
                reasons.append(f"cycle_{ci}_missing_or_fragmented_slide")
            before = q[q < s[0]] if len(s) else np.array([], dtype=int)
            after = q[q > s[-1]] if len(s) else np.array([], dtype=int)
            tail = mask & (r["cycle_id"] == ci) & (r["time"] >= end - PROTOCOL["a2"]["minimum_final_quasi_dwell_s"] - 1e-10)
            if not len(before) or not len(after) or not np.any(tail) or not np.all(r["mode"][tail] == 1):
                reasons.append(f"cycle_{ci}_missing_stable_quasi_return")
            if len(before) and len(after):
                result["first_slide_s"] = float(r["time"][idx[s[0]]])
                result["return_quasi_s"] = float(r["time"][idx[after[0]]])
                for name, left, right in [("quasi_to_slide", before[-1], s[0]), ("slide_to_quasi", s[-1], after[0])]:
                    events.append({"type": name, "cycle": ci, "time_interval_s": [float(r["time"][idx[left]]), float(r["time"][idx[right]])]})
        cycle_results.append(result)
    return {"accepted": not reasons, "reasons": sorted(set(reasons)), "cycles": cycle_results, "events": events}


def assess_a3(r, info):
    reasons = basic_reasons(r, info)
    mask = r["analysis_mask"]
    c = PROTOCOL["a3"]
    dt = info["signature"]["timestep"]
    kind = info["parameters"]["kind"]
    result_cycles, events = [], []
    if kind == "a3_contact_control":
        if np.any(mask & ~r["contact_flag"]) or not info["terminal"].get("contact_flag", False):
            reasons.append("contact_control_separated")
    elif kind == "a3_free_control":
        if np.any(mask & r["contact_flag"]) or info["terminal"].get("contact_flag", True):
            reasons.append("free_control_contacted")
    else:
        for ci, (start, end) in enumerate(info["program"]["cycle_windows"]):
            idx = np.flatnonzero(mask & (r["cycle_id"] == ci))
            intervals = runs_of(~r["contact_flag"][idx])
            result = {"cycle": ci, "separation_intervals": len(intervals)}
            if len(intervals) != 1:
                reasons.append(f"cycle_{ci}_separation_count")
            for a, b in intervals:
                if a == 0 or b >= len(idx):
                    reasons.append(f"cycle_{ci}_incomplete_separation")
                    continue
                exit_time, entry_time = float(r["time"][idx[a]]), float(r["time"][idx[b]])
                max_gap = float(np.max(r["gap"][idx[a:b]]))
                duration = (b-a)*dt
                contact_duration = float(np.sum(r["contact_flag"][idx]) * dt)
                impulse_mask = (r["time"] >= entry_time - 1e-10) & (r["time"] < entry_time + c["recontact_impulse_window_s"] - 1e-10)
                impulse = float(dt * np.sum(r["normal_force"][impulse_mask]))
                result.update(exit_s=exit_time, entry_s=entry_time, max_gap_m=max_gap,
                              free_duration_s=duration, free_samples=b-a,
                              contact_duration_s=contact_duration, recontact_impulse_Ns=impulse)
                if max_gap < c["minimum_max_gap_m"] or duration < c["minimum_free_duration_s"] or b-a < c["minimum_free_samples"]:
                    reasons.append(f"cycle_{ci}_separation_not_resolved")
                if contact_duration < c["minimum_contact_duration_s"]:
                    reasons.append(f"cycle_{ci}_insufficient_contact_dwell")
                events += [{"type": "contact_to_free", "cycle": ci, "time_s": exit_time},
                           {"type": "free_to_contact", "cycle": ci, "time_s": entry_time}]
            result_cycles.append(result)
    return {"accepted": not reasons, "reasons": sorted(set(reasons)), "cycles": result_cycles, "events": events}


def compare_a3(primary, primary_info, primary_qa, reference, reference_info, reference_qa):
    c = PROTOCOL["a3"]
    reasons = []
    comparisons = []
    if not reference_qa["accepted"]:
        reasons.append("reference_episode_failed_quality")
    if primary_info["parameters"]["kind"] == "a3":
        for left, right in zip(primary_qa["cycles"], reference_qa["cycles"]):
            if "exit_s" not in left or "exit_s" not in right:
                reasons.append("unmatched_refinement_events")
                continue
            metrics = {key: abs(left[key] - right[key]) for key in ["exit_s", "entry_s", "free_duration_s", "max_gap_m", "recontact_impulse_Ns"]}
            limits = {"exit_s": c["event_time_tolerance_s"], "entry_s": c["event_time_tolerance_s"],
                      "free_duration_s": c["free_duration_tolerance_s"],
                      "max_gap_m": max(c["gap_absolute_tolerance_m"], c["gap_relative_tolerance"] * abs(right["max_gap_m"])),
                      "recontact_impulse_Ns": max(c["impulse_absolute_tolerance_Ns"], c["impulse_relative_tolerance"] * abs(right["recontact_impulse_Ns"]))}
            failed = [key for key in metrics if metrics[key] > limits[key]]
            comparisons.append({"cycle": left["cycle"], "absolute_differences": metrics, "limits": limits, "failed_metrics": failed})
            reasons.extend(f"cycle_{left['cycle']}_refinement_{key}" for key in failed)
        if len(primary_qa["cycles"]) != len(reference_qa["cycles"]):
            reasons.append("refinement_cycle_count_mismatch")
    else:
        ids = np.flatnonzero(primary["analysis_mask"])
        j = np.rint(primary["time"][ids] / reference_info["signature"]["timestep"]).astype(int)
        if not len(ids) or not len(j) or j.max() >= len(reference["time"]):
            reasons.append("reference_missing_matched_states")
        else:
            qerr = float(np.max(abs(primary["qpos"][ids] - reference["qpos"][j])))
            verr = float(np.max(abs(primary["qvel"][ids] - reference["qvel"][j])))
            comparisons.append({"maximum_position_error_m": qerr, "maximum_velocity_error_m_s": verr})
            if qerr > c["same_mode_state_max_error_m"] or verr > c["same_mode_velocity_max_error_m_s"]:
                reasons.append("same_mode_refinement_state_error")
    return {"passed": not reasons, "reasons": sorted(set(reasons)), "comparisons": comparisons}


def save_trace(folder, r, info, qa, thresholds_hash=None, role="candidate"):
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder / "trace.npz", **r)
    write_json(folder / "episode.json", {**info, "quality": qa, "thresholds_sha256": thresholds_hash, "role": role})
    write_json(folder / "events.json", qa.get("events", []))


def a2_parameters(rng, mu, cycles, control=False, nominal_preload=None):
    c = PROTOCOL["a2"]
    result = {"kind": "a2_quasi_control" if control else "a2", "mu": mu,
              "preload": uniform(rng, c["preload_N"]) if nominal_preload is None else nominal_preload,
              "initial_y": uniform(rng, c["initial_y_m"]), "initial_low_ratio": uniform(rng, c["low_ratio"]), "cycles": []}
    for _ in range(cycles):
        result["cycles"].append({"low_ratio": uniform(rng, c["low_ratio"]),
                                 "high_ratio": uniform(rng, c["control_high_ratio"] if control else c["high_ratio"]),
                                 **{key: uniform(rng, c[key]) for key in ["rise_s", "high_hold_s", "fall_s", "low_hold_s"]}})
    return result


def nominal_a2(preload, mu):
    return {"kind": "a2", "mu": mu, "preload": preload, "initial_y": 0.0, "initial_low_ratio": 0.4,
            "cycles": [{"low_ratio": 0.4, "high_ratio": 1.04, "rise_s": 0.4,
                        "high_hold_s": 0.02, "fall_s": 0.08, "low_hold_s": 0.4}]}


def a3_parameters(rng, cycles, kind):
    c = PROTOCOL["a3"]
    preload = uniform(rng, c["nominal_preload_N"])
    initial_qx = Q_TOUCH
    center = Q_TOUCH + preload / c["stiffness_N_m"]
    amp = uniform(rng, c["amplitude_m"])
    frequency = uniform(rng, c["frequency_Hz"])
    if kind == "a3_contact_control":
        amp = uniform(rng, c["contact_control_amplitude_m"])
    elif kind == "a3_free_control":
        initial_qx = Q_TOUCH - c["free_control_initial_gap_m"]
        center = initial_qx
        amp = uniform(rng, c["free_control_amplitude_m"])
        frequency = uniform(rng, c["free_control_frequency_Hz"])
        preload = 0.0  # No nominal wall preload for free-space comparison episodes.
    return {"kind": kind, "nominal_preload": preload, "initial_qx": initial_qx,
            "initial_y": uniform(rng, c["initial_y_m"]), "support_center": center,
            "amplitude": amp, "frequency": frequency, "phase": float(rng.uniform(0, 2*math.pi)), "cycle_count": cycles}


def calibrate(model_path, dt, root):
    root.mkdir(parents=True, exist_ok=False)
    _, _, _, signature = model_info(model_path, dt)
    write_json(root / "protocol_before_calibration.json", PROTOCOL)
    write_json(root / "signature.json", signature)
    c = PROTOCOL["calibration"]
    summaries = []
    creep_max = 0.0
    failure_reasons = []
    for role, loads in [("fit", c["fit_preloads_N"]), ("validation", c["validation_preloads_N"])]:
        for load in loads:
            for ratio in c["subthreshold_ratios"]:
                parameters = {"kind": "calibration_hold", "preload": load, "ratio": ratio, "mu": signature["mu"], "initial_y": 0.0}
                r, info = simulate(model_path, dt, parameters)
                mask = r["analysis_mask"]
                reasons = basic_reasons(r, info)
                if np.any(mask & (~r["contact_flag"] | (r["normal_force"] <= c["loaded_force_N"]))):
                    reasons.append("subthreshold_contact_failure")
                if not np.any(mask) or not np.all(np.isfinite(r["rho"][mask])) or np.any(r["rho"][mask] >= c["rho_primary"]):
                    reasons.append("subthreshold_not_inside_cone")
                speed = abs(r["qvel"][mask, 1])
                maximum = float(np.max(speed)) if len(speed) else float("nan")
                if role == "fit" and not reasons:
                    creep_max = max(creep_max, maximum)
                name = f"{role}_preload_{load:g}_ratio_{ratio:g}"
                summary = {"name": name, "role": role, "preload_N": load, "ratio": ratio,
                           "maximum_speed_m_s": maximum, "median_speed_m_s": float(np.median(speed)) if len(speed) else None,
                           "accepted": not reasons, "reasons": reasons}
                summaries.append(summary)
                save_trace(root / "traces" / name, r, info, summary, role="calibration_only")
                if reasons:
                    failure_reasons.append(name)
        print(f"Calibration {role}: finished preloads {loads}", flush=True)
    if not math.isfinite(creep_max) or creep_max <= 0:
        failure_reasons.append("invalid_creep_envelope")
    quasi = c["quasi_ceiling_multiplier"] * creep_max
    thresholds = {"protocol_version": VERSION, "signature": signature, "source_script_sha256": digest_bytes(Path(__file__).read_bytes()),
                  "fit_creep_max_m_s": creep_max,
                  "primary": {"quasi_speed_m_s": quasi, "slide_speed_m_s": c["slide_floor_multiple_of_quasi"] * quasi,
                              "rho_boundary": c["rho_primary"], "loaded_force_N": c["loaded_force_N"]},
                  "sensitivity": {"quasi_speed_m_s": [x * creep_max for x in c["quasi_sensitivity_multipliers"]],
                                  "slide_speed_m_s": [x * quasi for x in c["slide_sensitivity_multiples_of_primary_quasi"]],
                                  "rho_boundary": c["rho_sensitivity"], "loaded_force_N": c["loaded_force_sensitivity_N"],
                                  "a3_max_gap_m": PROTOCOL["a3"]["gap_sensitivity_m"],
                                  "a3_free_duration_s": PROTOCOL["a3"]["free_duration_sensitivity_s"]},
                  "freeze_rule": "Computed from calibration only; validation never changes thresholds. Generation refuses a different protocol/model/version/timestep.",
                  "calibrated_preload_range_N": [min(c["fit_preloads_N"]), max(c["fit_preloads_N"])]}
    for summary in summaries:
        if summary["role"] == "validation" and (summary["maximum_speed_m_s"] is None or summary["maximum_speed_m_s"] > quasi):
            failure_reasons.append(summary["name"] + "_exceeds_frozen_quasi_ceiling")
    supra = []
    for load in sorted(c["fit_preloads_N"] + c["validation_preloads_N"]):
        parameters = nominal_a2(load, signature["mu"])
        r, info = simulate(model_path, dt, parameters)
        label_trace(r, thresholds, "a2")
        qa = assess_a2(r, info, thresholds)
        entry = {"preload_N": load, "peak_speed_m_s": float(np.max(abs(r["qvel"][:, 1]))), **qa}
        supra.append(entry)
        save_trace(root / "traces" / f"supra_validation_{load:g}", r, info, qa, role="calibration_only")
        if not qa["accepted"]:
            failure_reasons.append(f"supra_validation_{load:g}")
    report = {"passed": not failure_reasons, "failure_reasons": failure_reasons,
              "subthreshold_cases": summaries, "suprathreshold_validation": supra,
              "candidate_thresholds": thresholds}
    write_json(root / "calibration_report.json", report)
    if failure_reasons:
        raise RuntimeError(f"Calibration failed; all traces retained in {root}. No thresholds file was frozen. See calibration_report.json.")
    write_json(root / "thresholds.json", thresholds)
    print(f"Frozen thresholds: quasi <= {quasi:.9g} m/s; slide >= {thresholds['primary']['slide_speed_m_s']:.9g} m/s", flush=True)
    return root / "thresholds.json"


def planned_splits(count, rng):
    if count == 0:
        return []
    if count == 1:
        return ["train"]
    if count == 2:
        labels = ["train", "test"]
    else:
        nval = max(1, int(round(count * .15)))
        ntest = max(1, int(round(count * .15)))
        labels = ["train"] * (count - nval - ntest) + ["validation"] * nval + ["test"] * ntest
    return [labels[i] for i in rng.permutation(count)]


def persist_candidate(root, identifier, parameters, model_path, dt, thresholds, threshold_hash, split):
    primary, info = simulate(model_path, dt, parameters)
    label_trace(primary, thresholds, parameters["kind"])
    qa = assess_a2(primary, info, thresholds) if parameters["kind"].startswith("a2") else assess_a3(primary, info)
    folder = root / "episodes" / identifier
    pending = dict(qa)
    if parameters["kind"].startswith("a3"):
        pending.update(accepted=False, reasons=qa["reasons"] + ["refinement_pending"])
    # Persist the primary attempt before starting verification: even an unexpected
    # verification failure cannot erase the primary trajectory.
    save_trace(folder, primary, info, pending, threshold_hash, role="candidate")
    if parameters["kind"].startswith("a3"):
        reference, ref_info = simulate(model_path, dt * PROTOCOL["a3"]["reference_step_ratio"], parameters)
        label_trace(reference, thresholds, parameters["kind"])
        ref_qa = assess_a3(reference, ref_info)
        save_trace(root / "verification" / identifier, reference, ref_info, ref_qa, threshold_hash, role="verification_only")
        comparison = compare_a3(primary, info, qa, reference, ref_info, ref_qa)
        qa["refinement"] = comparison
        qa["reasons"] = sorted(set(qa["reasons"] + comparison["reasons"]))
        qa["accepted"] = not qa["reasons"]
    write_json(folder / "episode.json", {**info, "quality": qa, "thresholds_sha256": threshold_hash, "role": "candidate"})
    write_json(folder / "events.json", qa.get("events", []))
    eligible_count = int(np.sum(primary["analysis_mask"])) if qa["accepted"] else 0
    return {"id": identifier, "family": parameters["kind"], "split_planned": split,
            "accepted": qa["accepted"], "reasons": qa["reasons"],
            "trace": str((folder / "trace.npz").relative_to(root)),
            "metadata": str((folder / "episode.json").relative_to(root)),
            "raw_rows": len(primary["time"]), "eligible_rows": eligible_count,
            "analysis_window_s": [info["program"]["analysis_start"], info["program"]["analysis_end"]],
            "thresholds_sha256": threshold_hash}


def generate(model_path, dt, root, calibration_path, a2_count, a3_count, control_count, cycles, seed):
    if (root / "manifest.json").exists() or (root / "episodes").exists():
        raise FileExistsError("This output already has candidate episodes. Choose a new output directory; nothing will be overwritten.")
    thresholds = json.loads(calibration_path.read_text())
    _, _, _, signature = model_info(model_path, dt)
    if digest_bytes(Path(__file__).read_bytes()) != thresholds["source_script_sha256"]:
        raise ValueError("The generator code changed since calibration. Recalibrate in a new directory before generating data.")
    if signature != thresholds["signature"]:
        raise ValueError("Frozen calibration does not match this XML, MuJoCo version, timestep, or protocol. Run a new calibration in a new directory.")
    threshold_hash = digest_bytes(calibration_path.read_bytes())
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "frozen_thresholds.json", thresholds)
    # Keep the original file bytes as well so its recorded hash is reproducible.
    (root / "thresholds_original.json").write_bytes(calibration_path.read_bytes())
    write_json(root / "protocol.json", PROTOCOL)
    (root / "model.xml").write_bytes(model_path.read_bytes())
    schedule_rng = np.random.default_rng(seed)
    families = [("a2", a2_count), ("a3", a3_count), ("a2_quasi_control", control_count),
                ("a3_contact_control", control_count), ("a3_free_control", control_count)]
    plan = []
    for family, count in families:
        splits = planned_splits(count, schedule_rng)
        for index in range(count):
            episode_seed = int(schedule_rng.integers(0, 2**32))
            rng = np.random.default_rng(episode_seed)
            parameters = (a2_parameters(rng, signature["mu"], cycles, control=family != "a2")
                          if family.startswith("a2") else a3_parameters(rng, cycles, family))
            parameters["episode_seed"] = episode_seed
            plan.append({"id": f"{family}_{index:04d}", "split": splits[index], "parameters": parameters})
    write_json(root / "candidate_plan_before_simulation.json", {"seed": seed, "episodes": plan})
    manifest = {"version": VERSION, "seed": seed, "signature": signature, "thresholds_sha256": threshold_hash,
                "calibration_source": str(calibration_path.resolve()), "complete": False,
                "training_rule": "Use accepted episodes only, their analysis_mask & valid_transition rows, and the planned episode split. Never use calibration/verification traces as training examples.",
                "episodes": []}
    write_json(root / "manifest.json", manifest)
    for n, candidate in enumerate(plan):
        identifier, parameters = candidate["id"], candidate["parameters"]
        try:
            entry = persist_candidate(root, identifier, parameters, model_path, dt, thresholds, threshold_hash, candidate["split"])
        except Exception:
            folder = root / "episodes" / identifier
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "generation_error.json", {"parameters": parameters, "error": traceback.format_exc()})
            retained_trace = folder / "trace.npz"
            retained_count = 0
            if retained_trace.exists():
                with np.load(retained_trace) as retained:
                    retained_count = len(retained["time"])
            entry = {"id": identifier, "family": parameters["kind"], "split_planned": candidate["split"],
                     "accepted": False, "reasons": ["generation_exception"],
                     "trace": str(retained_trace.relative_to(root)) if retained_trace.exists() else None,
                     "metadata": str((folder / "generation_error.json").relative_to(root)),
                     "raw_rows": retained_count, "eligible_rows": 0}
        manifest["episodes"].append(entry)
        write_json(root / "manifest.json", manifest)
        print(f"[{n+1}/{len(plan)}] {identifier}: {'ACCEPT' if entry['accepted'] else 'REJECT'} {', '.join(entry['reasons'])}", flush=True)
    manifest["complete"] = True
    write_json(root / "manifest.json", manifest)
    accepted = [e for e in manifest["episodes"] if e["accepted"]]
    write_json(root / "accepted_splits.json", {split: [e["trace"] for e in accepted if e["split_planned"] == split] for split in ["train", "validation", "test"]})
    make_report(root, thresholds, manifest)
    return manifest


def make_report(root, thresholds, manifest):
    entries = manifest["episodes"]
    accepted = [e for e in entries if e["accepted"]]
    rejected = [e for e in entries if not e["accepted"]]
    c = thresholds["primary"]
    lines = ["# E1 data-generation report", "",
             f"Generated {len(entries)} candidates: **{len(accepted)} accepted**, **{len(rejected)} rejected**. No replacement sampling.", "",
             "## Frozen calibration", "",
             f"- MuJoCo: {manifest['signature']['mujoco_version']}; primary timestep: {manifest['signature']['timestep']} s.",
             f"- Quasi-stick ceiling: {c['quasi_speed_m_s']:.9g} m/s; clear-slide floor: {c['slide_speed_m_s']:.9g} m/s.",
             f"- Friction-utilization boundary: {c['rho_boundary']}; minimum loaded force: {c['loaded_force_N']} N.",
             "- Full sensitivity ranges: frozen_thresholds.json. Thresholds were fixed before candidate simulation and before any model fitting.", "",
             "## Acceptance by family", "", "| Family | Attempted | Accepted | Rejected |", "|---|---:|---:|---:|"]
    for family in sorted({e["family"] for e in entries}):
        group = [e for e in entries if e["family"] == family]
        passed = sum(e["accepted"] for e in group)
        lines.append(f"| {family} | {len(group)} | {passed} | {len(group)-passed} |")
    lines += ["", "## Rejected episodes", ""]
    lines += [f"- {e['id']}: {', '.join(e['reasons'])}. Files retained under episodes/." for e in rejected] or ["None in this batch. This is not a guarantee for every parameter combination."]
    lines += ["", "## Startup and data use", "",
              "All startup and rejected rows are retained. Training eligibility is the intersection of an accepted manifest entry, analysis_mask, and valid_transition. Splits were assigned by whole episode before simulation; all cycles in an episode stay together.", "",
              "Calibration traces and finer-step verification traces are excluded from accepted_splits.json. mode, rho, forces, cycle ID, and support phase are diagnostics; the mode-agnostic model uses qpos/qvel and actual ctrl to predict qpos_next/qvel_next.", "",
              "## A3 resolution checks", "",
              "Every A3 candidate (including same-mode controls) was rerun at half the primary timestep. Transition episodes require exactly one complete separation/recontact per retained drive cycle, gap >= 1 mm, free duration >= 20 ms with >= 40 samples, and contact dwell >= 50 ms. Matched exit/entry and duration must agree within 1 ms; peak gap within max(2%, 50 micrometres); the first 20 ms normal impulse after recontact within max(5%, 0.005 N s). Detailed comparisons are in each episode.json.", "",
              "## Limits", "",
              "These are operational labels and numerical acceptance tests, not physical validation of a marker. E1b matching/common-support checks, model capacity checks, and prediction-error analysis remain subsequent work. The included same-mode control families do not guarantee matched state changes for every transition. Rejected parameter regions must be reported when interpreting the accepted dataset."]
    (root / "DATA_REPORT.md").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        family_names = sorted({e["family"] for e in entries})
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        passed = [sum(e["accepted"] for e in entries if e["family"] == f) for f in family_names]
        failed = [sum(not e["accepted"] for e in entries if e["family"] == f) for f in family_names]
        x = np.arange(len(family_names))
        ax.bar(x, passed, label="Accepted", color="#0f766e")
        ax.bar(x, failed, bottom=passed, label="Rejected and retained", color="#b45309")
        ax.set_xticks(x, [f.replace("_", "\n") for f in family_names])
        ax.set_ylabel("Episodes")
        ax.set_title("E1 generation: all attempts accounted for")
        ax.legend()
        fig.savefig(root / "acceptance.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent.parent / "mujoco_contact" / "models" / "wall_contact.xml")
    parser.add_argument("--output", type=Path, default=Path("e1_dataset"))
    parser.add_argument("--stage", choices=["all", "calibrate", "generate"], default="all")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--timestep", type=float, default=0.000125)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--a2-episodes", type=int, default=8)
    parser.add_argument("--a3-episodes", type=int, default=6)
    parser.add_argument("--control-episodes", type=int, default=3, help="Count for EACH of three same-mode control families.")
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    if not math.isfinite(args.timestep) or args.timestep <= 0:
        parser.error("--timestep must be finite and positive")
    if args.cycles < 1 or any(x < 0 for x in [args.a2_episodes, args.a3_episodes, args.control_episodes]):
        parser.error("Cycle count must be positive; episode counts must be nonnegative")
    if args.stage in ["all", "calibrate"] and args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new empty output directory for calibration; existing results are never overwritten.")
    start = time.perf_counter()
    if args.stage in ["all", "calibrate"]:
        calibration_path = calibrate(args.model, args.timestep, args.output / "calibration")
    else:
        if args.calibration is None:
            parser.error("--calibration is required with --stage generate")
        calibration_path = args.calibration
    if args.stage in ["all", "generate"]:
        manifest = generate(args.model, args.timestep, args.output, calibration_path,
                            args.a2_episodes, args.a3_episodes, args.control_episodes, args.cycles, args.seed)
        passed = sum(e["accepted"] for e in manifest["episodes"])
        print(f"Finished: {passed}/{len(manifest['episodes'])} accepted. Rejected episodes retained. Report: {args.output / 'DATA_REPORT.md'}", flush=True)
    print(f"Elapsed: {time.perf_counter()-start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
