"""Normal-only A3: a driven spring-damper unloads and reloads the sphere.

For testing whether learned dynamics during repeated contact transitions under known excitation

With viewer:
    python a3_normal_spring_damper.py
Without viewer:
    python a3_normal_spring_damper.py --headless --no-show
Refinement example:
    python a3_normal_spring_damper.py --headless --no-show --timestep 0.00025
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import mujoco
import numpy as np


# Initial calibration parameters, not measured marker material properties.
Q_TOUCH = 2.75 # Exact touching joint coordinate for the given XML.
STIFFNESS = 1000.0 # N/m
DAMPING = 30.0 # N s/m
NOMINAL_PRELOAD = 10.0 # N; actual settled load differs due to soft contact.
SUPPORT_REST = Q_TOUCH + NOMINAL_PRELOAD / STIFFNESS
AMPLITUDE = 0.020 # m: sinusoidal support amplitude about SUPPORT_REST
FREQUENCY = 2.0 # Hz
SETTLE_TIME = 0.3 # s: stationary support before excitation
FADE_TIME = 0.5 # s: smoothly increase excitation amplitude


def support_motion(t):
    """Return support position and its analytic time derivative.

    A quintic envelope starts with zero velocity and acceleration and reaches
    unit amplitude smoothly. Including the envelope derivative in support_vel
    is essential for the damping force.

    F = k * (x_support - x_sphere) + c * (v_support - v_sphere)
    """
    tau = t - SETTLE_TIME
    if tau <= 0.0:
        return SUPPORT_REST, 0.0
    r = min(tau / FADE_TIME, 1.0)
    envelope = 10.0 * r**3 - 15.0 * r**4 + 6.0 * r**5
    envelope_rate = 30.0 * r**2 * (1.0 - r)**2 / FADE_TIME if r < 1.0 else 0.0
    omega = 2.0 * np.pi * FREQUENCY
    phase = omega * tau
    position = SUPPORT_REST + AMPLITUDE * envelope * np.sin(phase)
    velocity = AMPLITUDE * (
        envelope_rate * np.sin(phase) + envelope * omega * np.cos(phase)
    )
    return position, velocity


def vector_text(values):
    return "[" + " ".join(format(float(v), ".17g") for v in values) + "]"


def make_plot(records, destination, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = records["time"]
    fig, axs = plt.subplots(5, 1, figsize=(10, 11), sharex=True, layout="constrained")
    axs[0].plot(t, 1000 * (records["support_pos"] - Q_TOUCH), label="Support reference")
    axs[0].plot(t, 1000 * (records["qpos"][:, 0] - Q_TOUCH), label="Sphere")
    axs[0].axhline(0, color="gray", linestyle=":", linewidth=1)
    axs[0].set_ylabel("Normal position\nrelative to touch (mm)")
    axs[0].legend(loc="upper right")

    axs[1].plot(t, 1000 * records["gap"], color="tab:purple")
    axs[1].axhline(0, color="gray", linestyle=":", linewidth=1)
    axs[1].set_ylabel("Signed surface gap (mm)\npositive = separation")

    axs[2].plot(t, records["qvel"][:, 0], color="tab:blue")
    axs[2].axhline(0, color="gray", linestyle=":", linewidth=1)
    axs[2].set_ylabel("Normal velocity (m/s)")

    axs[3].plot(t, records["ctrl"][:, 0], label="Applied support force Fx")
    axs[3].plot(t, records["normal_force"], label="Contact normal magnitude Fn")
    axs[3].set_ylabel("Force (N)")
    axs[3].legend(loc="upper right")

    axs[4].step(t, records["contact_flag"].astype(int), where="post")
    axs[4].set_yticks([0, 1], ["No contact", "Contact"])
    axs[4].set_ylim(-0.15, 1.15)
    axs[4].set_xlabel("Simulation time (s)")
    for ax in axs:
        ax.axvspan(0, SETTLE_TIME, color="gray", alpha=0.10)
        ax.grid(alpha=0.2)
    fig.suptitle("A3 normal-only: driven spring-damper and wall contact")
    fig.savefig(destination, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("mujoco_contact/models/wall_contact.xml"))
    parser.add_argument("--output-dir", type=Path, default=Path("mujoco_contact/a3_normal"))
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--timestep", type=float, help="Optional runtime override; XML file is not changed.")
    parser.add_argument("--headless", action="store_true", help="Run without the passive viewer.")
    parser.add_argument("--no-show", action="store_true", help="Save the plot without opening its window.")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if args.timestep is not None and (not np.isfinite(args.timestep) or args.timestep <= 0):
        parser.error("--timestep must be finite and positive")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    if args.timestep is not None:
        model.opt.timestep = args.timestep
    if (model.nq, model.nv, model.nu) != (2, 2, 2):
        raise ValueError("This script expects the provided two-joint, two-motor XML.")
    wall_id = model.geom("wall_geom").id
    sphere_id = model.geom("object_geom").id
    data = mujoco.MjData(model)
    data.qpos[:] = [Q_TOUCH, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    n_steps = int(np.ceil(args.duration / dt))
    scalar_fields = ["time", "time_next", "support_pos", "support_vel", "spring_force",
                     "damping_force", "normal_force", "tangential_force", "gap",
                     "contact_distance", "rho"]
    records = {name: np.empty(n_steps, dtype=float) for name in scalar_fields}
    for name in ["qpos", "qvel", "qacc", "ctrl", "qpos_next", "qvel_next"]:
        records[name] = np.empty((n_steps, 2), dtype=float)
    records["contact_flag"] = np.empty(n_steps, dtype=bool)
    records["mode"] = np.empty(n_steps, dtype="U16")
    records["phase"] = np.empty(n_steps, dtype="U16")
    records["solver_iterations"] = np.empty(n_steps, dtype=int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "a3_normal_dt_" + format(dt, ".9f").rstrip("0").replace(".", "p")
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
            support_pos, support_vel = support_motion(t)
            spring_force = STIFFNESS * (support_pos - data.qpos[0])
            damping_force = DAMPING * (support_vel - data.qvel[0])
            data.ctrl[0] = spring_force + damping_force
            data.ctrl[1] = 0.0  # First A3 milestone: validate normal motion alone.

            # All logged state/contact quantities refer to the same pre-step state.
            # The support force is recomputed at every simulation step.
            mujoco.mj_forward(model, data)
            fn = ft = 0.0
            distances = []
            for j in range(data.ncon):
                contact = data.contact[j]
                if {int(contact.geom1), int(contact.geom2)} != {wall_id, sphere_id}:
                    continue
                distances.append(float(contact.dist))
                mujoco.mj_contactForce(model, data, j, contact_force)
                fn += float(contact_force[0])
                ft += float(np.linalg.norm(contact_force[1:3]))
            contact_flag = bool(distances)
            contact_distance = min(distances) if distances else np.nan
            # True signed geom distance remains available when no contact exists.
            gap = float(mujoco.mj_geomDistance(model, data, wall_id, sphere_id, 100.0, None))
            rho = ft / (0.5 * fn) if fn > 1e-8 else np.nan
            mode = "FREE" if not contact_flag else ("BOUNDARY" if fn <= 0.1 else "CONTACT")
            phase = "SETTLE" if t < SETTLE_TIME else ("FADE_IN" if t < SETTLE_TIME + FADE_TIME else "PERIODIC")
            values = dict(time=t, support_pos=support_pos, support_vel=support_vel,
                          spring_force=spring_force, damping_force=damping_force,
                          normal_force=fn, tangential_force=ft, gap=gap,
                          contact_distance=contact_distance, rho=rho,
                          contact_flag=contact_flag, mode=mode, phase=phase,
                          solver_iterations=int(np.max(data.solver_niter)),
                          qpos=data.qpos, qvel=data.qvel, qacc=data.qacc, ctrl=data.ctrl)
            for name, value in values.items():
                records[name][i] = value
            log.write(
                f"time={t:.9f}, qpos={vector_text(data.qpos)}, qvel={vector_text(data.qvel)}, "
                f"normal_force={fn!r}, tangential_force={ft!r}, Fctrl={vector_text(data.ctrl)}, "
                f"support_pos={support_pos!r}, support_vel={support_vel!r}, "
                f"spring_force={spring_force!r}, damping_force={damping_force!r}, "
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
    np.savez_compressed(prefix.with_suffix(".npz"), **records)
    metadata = dict(mujoco_version=mujoco.__version__, model_path=str(args.model.resolve()),
                    model_xml=args.model.read_text(), timestep=dt, steps=written,
                    requested_duration=args.duration, actual_duration=float(data.time),
                    stiffness=STIFFNESS, damping=DAMPING, nominal_preload=NOMINAL_PRELOAD,
                    support_rest=SUPPORT_REST, amplitude=AMPLITUDE, frequency=FREQUENCY,
                    settle_time=SETTLE_TIME, fade_time=FADE_TIME, q_touch=Q_TOUCH,
                    integrator=int(model.opt.integrator), solver=int(model.opt.solver),
                    solver_iteration_limit=int(model.opt.iterations), tolerance=float(model.opt.tolerance),
                    impratio=float(model.opt.impratio), geom_solref=model.geom_solref.tolist(),
                    geom_solimp=model.geom_solimp.tolist(),
                    warning_counts=data.warning.number.tolist(),
                    logging="State/contact quantities at t; ctrl held over [t, t_next]; next state saved separately.",
                    labels="Normal-only raw labels; FREE is not a validated macroscopic-separation label.")
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    make_plot(records, prefix.with_suffix(".png"), show=not args.no_show)
    print(f"Saved {written} samples, dt={dt:g} s, simulated duration={data.time:.6f} s")
    print(f"Maximum sampled gap: {1000 * np.max(records['gap']):.3f} mm")
    print(f"Peak normal contact force: {np.max(records['normal_force']):.3f} N")
    print(f"Outputs: {prefix}.*")


if __name__ == "__main__":
    main()
