"""A2: quasi-stick -> slide under constant applied normal preload.

  viewer: python a2_quasi_stick_slide.py
  No windows:       python a2_quasi_stick_slide.py --headless --no-show
  Refine timestep:  python a2_quasi_stick_slide.py --headless --no-show --timestep 0.00025
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import mujoco
import numpy as np


Q_TOUCH = 2.75  # Joint coordinate for exact touching in the supplied wall XML.
SETTLE_TIME = 0.3
RHO_BOUNDARY = 0.98
QUASI_STICK_SPEED = 5e-4   # m/s: provisional ceiling from the calibration trace
SLIDE_SPEED = 2e-3         # m/s: provisional clear-slide threshold
MIN_LOADED_FORCE = 0.1    # N


def vector_text(values):
    return "[" + " ".join(format(float(v), ".17g") for v in values) + "]"


def classify(contact_flag, fn, rho, vt):
    if not contact_flag:
        return "FREE"
    if fn <= MIN_LOADED_FORCE:
        return "BOUNDARY"
    if rho < RHO_BOUNDARY and vt <= QUASI_STICK_SPEED:
        return "QUASI_STICK"
    if rho >= RHO_BOUNDARY and vt >= SLIDE_SPEED:
        return "SLIDE"
    return "BOUNDARY"


def make_plot(r, destination, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = r["time"]
    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True, layout="constrained")
    axs[0].plot(t, 1000 * r["gap"], color="tab:purple")
    axs[0].axhline(0, color="gray", ls=":", lw=1)
    axs[0].set_ylabel("Signed surface gap (mm)\npositive = separation")

    axs[1].plot(t, r["qvel"][:, 1], color="tab:green", label="Tangential velocity vy")
    axs[1].axhline(QUASI_STICK_SPEED, color="gray", ls=":", label="Quasi-stick ceiling")
    axs[1].axhline(SLIDE_SPEED, color="tab:orange", ls="--", label="Clear-slide threshold")
    axs[1].set_ylabel("Tangential velocity (m/s)")
    axs[1].legend(loc="upper left", fontsize=8)

    axs[2].plot(t, r["ctrl"][:, 0], ls="--", label="Applied Fx")
    axs[2].plot(t, r["normal_force"], label="Contact Fn")
    axs[2].set_ylabel("Normal forces (N)")
    axs[2].legend(loc="lower right", fontsize=8)

    axs[3].plot(t, r["ctrl"][:, 1], ls="--", label="Applied Fy")
    axs[3].plot(t, r["tangential_force"], label="Contact |Ft|")
    axs[3].set_ylabel("Tangential forces (N)")
    axs[3].legend(loc="upper left", fontsize=8)

    axs[4].plot(t, r["rho"], label="Friction utilization")
    axs[4].axhline(RHO_BOUNDARY, color="gray", ls="--", label="Boundary threshold")
    axs[4].axhline(1, color="black", ls=":", lw=1)
    axs[4].set_ylabel("rho = |Ft| / (mu Fn)")
    axs[4].legend(loc="upper left", fontsize=8)

    label_codes = {"FREE": 0, "QUASI_STICK": 1, "BOUNDARY": 2, "SLIDE": 3}
    axs[5].step(t, [label_codes[m] for m in r["mode"]], where="post")
    axs[5].set_yticks([0, 1, 2, 3], ["Free", "Quasi-stick", "Boundary", "Slide"])
    axs[5].set_ylim(-0.2, 3.2)
    axs[5].set_xlabel("Simulation time (s)")
    for ax in axs:
        ax.axvspan(0, SETTLE_TIME, color="gray", alpha=0.10)
        ax.grid(alpha=0.2)
    fig.suptitle("A2: quasi-stick to slide during continuous contact")
    fig.savefig(destination, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("mujoco_contact/models/wall_contact.xml"))
    parser.add_argument("--output-dir", type=Path, default=Path("mujoco_contact/a2_slide"))
    parser.add_argument("--duration", type=float, default=2.95, help="Simulated seconds, not step count.")
    parser.add_argument("--timestep", type=float, help="Optional runtime override; does not edit the XML.")
    parser.add_argument("--normal-force", type=float, default=10.0)
    parser.add_argument("--ramp-rate", type=float, default=2.0, help="Tangential ramp rate in N/s.")
    parser.add_argument("--force-cap", type=float, default=5.5, help="Tangential motor force cap in N.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-show", action="store_true", help="Save the plot without opening a window.")
    args = parser.parse_args()
    for name in ["duration", "normal_force", "ramp_rate", "force_cap", "timestep"]:
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error("--" + name.replace("_", "-") + " must be finite and positive")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    if args.timestep is not None:
        model.opt.timestep = args.timestep
    if (model.nq, model.nv, model.nu) != (2, 2, 2):
        raise ValueError("This script expects the supplied two-joint, two-motor wall XML.")
    wall_id = model.geom("wall_geom").id
    sphere_id = model.geom("object_geom").id
    data = mujoco.MjData(model)
    data.qpos[:] = [Q_TOUCH, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    n_steps = int(np.ceil(args.duration / dt))
    scalars = ["time", "time_next", "normal_force", "tangential_force", "gap",
               "contact_distance", "rho", "mu", "normal_velocity", "tangential_speed"]
    records = {name: np.empty(n_steps, dtype=float) for name in scalars}
    for name in ["qpos", "qvel", "qacc", "ctrl", "qpos_next", "qvel_next"]:
        records[name] = np.empty((n_steps, 2), dtype=float)
    for name in ["contact_flag", "post_settle"]:
        records[name] = np.empty(n_steps, dtype=bool)
    for name in ["mode", "phase"]:
        records[name] = np.empty(n_steps, dtype="U16")
    records["solver_iterations"] = np.empty(n_steps, dtype=int)

    def tag(value):
        return format(value, ".9g").replace(".", "p")
    stem = (f"a2_dt_{tag(dt)}_fx_{tag(args.normal_force)}_ramp_{tag(args.ramp_rate)}"
            f"_cap_{tag(args.force_cap)}_T_{tag(args.duration)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / stem
    if args.headless:
        viewer_context = nullcontext(None)
    else:
        from mujoco import viewer as mujoco_viewer
        viewer_context = mujoco_viewer.launch_passive(model, data)

    written = 0
    contact_force = np.zeros(6)
    with prefix.with_suffix(".txt").open("w") as log, viewer_context as viewer:
        start_wall_time = time.perf_counter()
        next_viewer_time = 0.0
        for i in range(n_steps):
            if viewer is not None and not viewer.is_running():
                break
            t = float(data.time)
            data.ctrl[0] = args.normal_force
            data.ctrl[1] = min(args.ramp_rate * max(0.0, t - SETTLE_TIME), args.force_cap)

            # Recompute quantities at the current state with the current action.
            # Record them BEFORE integration, then save the resulting next state.
            mujoco.mj_forward(model, data)
            fn = ft = friction_capacity = 0.0
            distances = []
            mus = []
            for j in range(data.ncon):
                contact = data.contact[j]
                if {int(contact.geom1), int(contact.geom2)} != {wall_id, sphere_id}:
                    continue
                distances.append(float(contact.dist))
                mu = float(contact.friction[0])
                if not np.isclose(mu, contact.friction[1]):
                    raise ValueError("This utilization calculation assumes isotropic sliding friction.")
                mus.append(mu)
                mujoco.mj_contactForce(model, data, j, contact_force)
                fn += float(contact_force[0])
                ft += float(np.linalg.norm(contact_force[1:3]))
                friction_capacity += mu * float(contact_force[0])
            contact_flag = bool(distances)
            contact_distance = min(distances) if distances else np.nan
            gap = float(mujoco.mj_geomDistance(model, data, wall_id, sphere_id, 100.0, None))
            rho = ft / friction_capacity if friction_capacity > 1e-8 else np.nan
            vt = abs(float(data.qvel[1]))
            mode = classify(contact_flag, fn, rho, vt)
            post_settle = t >= SETTLE_TIME
            phase = "SETTLE" if not post_settle else ("HOLD" if data.ctrl[1] >= args.force_cap else "RAMP")
            values = dict(time=t, normal_force=fn, tangential_force=ft, gap=gap,
                          contact_distance=contact_distance, rho=rho,
                          mu=mus[0] if mus else np.nan, normal_velocity=float(data.qvel[0]),
                          tangential_speed=vt, contact_flag=contact_flag,
                          post_settle=post_settle, mode=mode, phase=phase,
                          solver_iterations=int(np.max(data.solver_niter)),
                          qpos=data.qpos, qvel=data.qvel, qacc=data.qacc, ctrl=data.ctrl)
            for name, value in values.items():
                records[name][i] = value
            log.write(
                f"time={t:.9f}, qpos={vector_text(data.qpos)}, qvel={vector_text(data.qvel)}, "
                f"normal_force={fn!r}, tangential_force={ft!r}, Fctrl={vector_text(data.ctrl)}, "
                f"contact_flag={contact_flag}, gap={gap!r}, contact_distance={contact_distance!r}, "
                f"rho={rho!r}, mode={mode}, phase={phase}\n"
            )
            mujoco.mj_step(model, data)
            records["time_next"][i] = data.time
            records["qpos_next"][i] = data.qpos
            records["qvel_next"][i] = data.qvel
            written = i + 1
            if viewer is not None and data.time >= next_viewer_time:
                viewer.sync()
                next_viewer_time = float(data.time) + 1.0 / 60.0
                delay = float(data.time) - (time.perf_counter() - start_wall_time)
                if delay > 0:
                    time.sleep(delay)

    if written == 0:
        raise RuntimeError("Viewer closed before any samples were recorded.")
    records = {name: value[:written] for name, value in records.items()}
    # Check the terminal state too: it is a next-state target in the dataset.
    mujoco.mj_forward(model, data)  # Current final held action; no additional step.
    terminal_gap = float(mujoco.mj_geomDistance(model, data, wall_id, sphere_id, 100.0, None))
    terminal_contact = any(
        {int(data.contact[j].geom1), int(data.contact[j].geom2)} == {wall_id, sphere_id}
        for j in range(data.ncon)
    )
    eligible = records["post_settle"]
    losses = np.flatnonzero(eligible & ~records["contact_flag"])
    first_loss = float(records["time"][losses[0]]) if len(losses) else (
        float(data.time) if data.time >= SETTLE_TIME and not terminal_contact else None
    )
    def first_time(label):
        indices = np.flatnonzero(eligible & (records["mode"] == label))
        return float(records["time"][indices[0]]) if len(indices) else None
    diagnostics = dict(first_contact_loss_after_settling=first_loss,
                       first_boundary_after_settling=first_time("BOUNDARY"),
                       first_slide_after_settling=first_time("SLIDE"),
                       terminal_contact=bool(terminal_contact), terminal_gap=terminal_gap,
                       has_post_settle_samples=bool(np.any(eligible)),
                       minimum_post_settle_Fn=float(np.min(records["normal_force"][eligible])) if np.any(eligible) else None)
    np.savez_compressed(prefix.with_suffix(".npz"), **records)
    metadata = dict(scenario="A2", mujoco_version=mujoco.__version__,
                    model_path=str(args.model.resolve()), model_xml=args.model.read_text(),
                    timestep=dt, steps=written, requested_duration=args.duration,
                    actual_duration=float(data.time), normal_force=args.normal_force,
                    ramp_rate=args.ramp_rate, force_cap=args.force_cap, settle_time=SETTLE_TIME,
                    q_touch=Q_TOUCH, rho_boundary=RHO_BOUNDARY,
                    quasi_stick_speed=QUASI_STICK_SPEED, slide_speed=SLIDE_SPEED,
                    minimum_loaded_force=MIN_LOADED_FORCE,
                    integrator=int(model.opt.integrator), solver=int(model.opt.solver),
                    solver_iteration_limit=int(model.opt.iterations), tolerance=float(model.opt.tolerance),
                    impratio=float(model.opt.impratio), geom_solref=model.geom_solref.tolist(),
                    geom_solimp=model.geom_solimp.tolist(), warning_counts=data.warning.number.tolist(),
                    diagnostics=diagnostics,
                    logging="State/contact quantities at t; ctrl held over [t, time_next]; next state saved separately.",
                    e1_note="Use post_settle to identify startup exclusion. Keep diagnostics/labels separate from baseline inputs. No automatic cropping on contact loss.")
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    make_plot(records, prefix.with_suffix(".png"), show=not args.no_show)
    print(f"Saved {written} samples, dt={dt:g} s, simulated duration={data.time:.6f} s")
    print(f"First SLIDE label after settling: {diagnostics['first_slide_after_settling']}")
    if first_loss is not None:
        print(f"A2 condition not maintained: contact loss at {first_loss:.9f} s. All data retained for diagnosis.")
    else:
        print("No contact loss detected after settling, including the terminal state.")
    print(f"Outputs: {prefix}.*")


if __name__ == "__main__":
    main()
