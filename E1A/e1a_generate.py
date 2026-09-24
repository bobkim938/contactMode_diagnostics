"""Generate paired A3 datasets; no training and no A2 threshold calibration.

From the project root: python E1A/e1a_generate.py --output run_001
Defaults: 20 pairs per family, 3 retained cycles, dt=0.000125 s.
Each pair shares its initial conditions, support drive and dataset split.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
BACKEND_PATH = HERE.parent / "E1" / "e1_generate.py"
spec = importlib.util.spec_from_file_location("e1a_backend", BACKEND_PATH)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)

CONDITIONS = {
    "original": [0.9, 0.95, 0.001, 0.5, 2.0],
    "modified": [0.0, 0.95, 0.001, 0.5, 2.0],
}
FAMILIES = ("a3", "a3_contact_control", "a3_free_control")
SPLITS = ("train", "validation", "test")
LABELS = {
    "primary": {"loaded_force_N": 0.1},
    "sensitivity": {"loaded_force_N": [0.05, 0.1, 0.2]},
    "mode_codes": {"FREE": 0, "BOUNDARY": 2, "CONTACT": 4},
    "note": "Fixed A3 force threshold, not calibrated A2 stick/slide thresholds.",
}


def make_plan(seed, count, cycles):
    """Split whole paired episodes before observing simulation outcomes."""
    rng = np.random.default_rng(seed)
    plan = []
    for family in FAMILIES:
        splits = backend.planned_splits(count, rng)
        for index, split in enumerate(splits):
            episode_seed = int(rng.integers(0, 2**32))
            parameters = backend.a3_parameters(np.random.default_rng(episode_seed), cycles, family)
            parameters["episode_seed"] = episode_seed
            plan.append({"id": f"{family}_{index:04d}", "family": family,
                         "split": split, "parameters": parameters})
    return plan


def write_model(source, destination, solimp, dt):
    """Set both contacting geoms and verify the effective mixed contact."""
    tree = ET.parse(source)
    root = tree.getroot()
    if root.find("include") is not None or root.find("asset") is not None:
        raise ValueError("Use the self-contained sphere/wall XML without assets/includes.")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", repr(dt))
    for name in ("wall_geom", "object_geom"):
        geom = root.find(f".//geom[@name='{name}']")
        if geom is None:
            raise ValueError(f"Missing geom {name}")
        geom.set("solimp", " ".join(map(str, solimp)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")
    model, wall, sphere, signature = backend.model_info(destination, dt)
    data = mujoco.MjData(model)
    data.qpos[0] = backend.Q_TOUCH + 1e-6
    data.qvel[0] = 0.3
    data.ctrl[0] = 10.0
    mujoco.mj_forward(model, data)
    contacts = [data.contact[i] for i in range(data.ncon)
                if set(data.contact[i].geom) == {wall, sphere}]
    if len(contacts) != 1 or not np.allclose(contacts[0].solimp, solimp, rtol=0, atol=1e-14):
        raise ValueError("Requested solimp does not match effective wall/object contact.")
    return {"signature": signature, "requested_solimp": solimp,
            "effective_contact_solimp": contacts[0].solimp.copy(),
            "effective_contact_solref": contacts[0].solref.copy(),
            "internal_impedance_floor": float(mujoco.mjMINIMP)}


def save_force_events(folder):
    """Keep force-threshold crossings distinct from geometric contact events."""
    with np.load(folder / "trace.npz") as trace:
        events = []
        for threshold in LABELS["sensitivity"]["loaded_force_N"]:
            loaded = trace["contact_flag"] & (trace["normal_force"] > threshold)
            for i in np.flatnonzero(loaded[1:] != loaded[:-1]) + 1:
                events.append({
                    "type": "loaded_onset" if loaded[i] else "loaded_offset",
                    "threshold_N": threshold, "row_index": int(i),
                    "time_s": float(trace["time"][i]),
                    "time_bracket_s": trace["time"][i-1:i+1].tolist(),
                    "analysis_eligible": bool(trace["analysis_mask"][i]
                                              and trace["analysis_mask"][i-1]),
                })
    backend.write_json(folder / "loaded_events.json", events)


def generate_member(root, candidate, dt, label_hash):
    """Retain every attempt, including failures during refinement."""
    identifier = candidate["id"]
    folder = root / "episodes" / identifier
    try:
        entry = backend.persist_candidate(
            root, identifier, candidate["parameters"], root / "model.xml", dt,
            LABELS, label_hash, candidate["split"]
        )
        for trace_folder in (folder, root / "verification" / identifier):
            save_force_events(trace_folder)
    except Exception:
        backend.write_json(folder / "generation_error.json", {
            "parameters": candidate["parameters"], "error": traceback.format_exc()
        })
        entry = {"id": identifier, "family": candidate["family"],
                 "split_planned": candidate["split"], "accepted": False,
                 "reasons": ["generation_exception"], "eligible_rows": 0,
                 "trace": f"episodes/{identifier}/trace.npz"}
    entry["member_accepted"] = entry.pop("accepted")
    entry["member_reasons"] = entry.pop("reasons")
    return entry


def finalize_pair(candidate, members):
    accepted = all(member["member_accepted"] for member in members.values())
    reasons = [f"{condition}:{reason}" for condition, member in members.items()
               for reason in member["member_reasons"]]
    for member in members.values():
        member["accepted"] = accepted
        member["reasons"] = reasons
        if not accepted:
            member["eligible_rows"] = 0
    return {"id": candidate["id"], "family": candidate["family"],
            "split": candidate["split"], "accepted": accepted,
            "reasons": reasons, "members": members}


def publish(root, manifest, complete):
    """Only completed accepted pairs enter identical per-condition splits."""
    pairs = manifest["pairs"]
    split_ids = {split: [p["id"] for p in pairs if p["accepted"] and p["split"] == split]
                 for split in SPLITS}
    counts = {family: {split: sum(p["accepted"] and p["family"] == family
                                 and p["split"] == split for p in pairs)
                       for split in SPLITS} for family in FAMILIES}
    manifest.update(complete=complete, accepted_pairs=sum(p["accepted"] for p in pairs),
                    counts=counts, training_ready=bool(complete and not manifest["smoke"]
                    and all(n > 0 for family in counts.values() for n in family.values())))
    backend.write_json(root / "pair_manifest.json", manifest)
    backend.write_json(root / "paired_splits.json", split_ids)
    for condition in CONDITIONS:
        backend.write_json(root / condition / "accepted_splits.json", {
            split: [f"episodes/{identifier}/trace.npz" for identifier in ids]
            for split, ids in split_ids.items()
        })
        backend.write_json(root / condition / "manifest.json", {
            "condition": condition, "complete": complete,
            "training_ready": manifest["training_ready"],
            "protocol_sha256": manifest["protocol_sha256"],
            "episodes": [p["members"][condition] for p in pairs],
        })
    lines = ["# E1A generation report", "",
             f"Complete: {complete}. Training ready: {manifest['training_ready']}.",
             f"Processed {len(pairs)}/{manifest['planned_pairs']} pairs; "
             f"accepted {manifest['accepted_pairs']}; rejected {len(pairs)-manifest['accepted_pairs']}.",
             "No rejected candidate was replaced. Both members of a rejected pair are excluded.", "",
             "| Family | Train pairs | Validation pairs | Test pairs |",
             "|---|---:|---:|---:|"]
    lines += [f"| {family} | {c['train']} | {c['validation']} | {c['test']} |"
              for family, c in counts.items()]
    lines += ["", "## Failed pairs", ""]
    lines += [f"- {p['id']} ({p['split']}): {', '.join(p['reasons'])}"
              for p in pairs if not p["accepted"]] or ["None."]
    (root / "DATA_REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="run_001", help="Relative to E1A, or an absolute path.")
    parser.add_argument("--model", type=Path, default=HERE.parent / "E1/run_001/model.xml")
    parser.add_argument("--pairs-per-family", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--timestep", type=float, default=0.000125)
    parser.add_argument("--smoke", action="store_true", help="3 pairs/family, 1 cycle; never marked training-ready.")
    args = parser.parse_args()
    if mujoco.__version__ != "3.12.0":
        parser.error(f"Use MuJoCo 3.12.0 for this protocol; found {mujoco.__version__}.")
    if args.smoke:
        args.pairs_per_family, args.cycles = 3, 1
    if args.pairs_per_family < 3 or args.cycles < 1 or not np.isfinite(args.timestep) or args.timestep <= 0:
        parser.error("Require at least 3 pairs/family, at least 1 cycle, and positive finite timestep.")
    output = Path(args.output).expanduser()
    root = (output if output.is_absolute() else HERE / output).resolve()
    root.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier study.
    sources = root / "sources"
    sources.mkdir()
    for path in (Path(__file__), BACKEND_PATH):
        (sources / path.name).write_bytes(path.read_bytes())
    (sources / "input_model.xml").write_bytes(args.model.read_bytes())
    model_records = {condition: write_model(args.model, root / condition / "model.xml", solimp, args.timestep)
                     for condition, solimp in CONDITIONS.items()}
    protocol = {
        "name": "E1A", "version": "1.0", "seed": args.seed,
        "timestep_s": args.timestep, "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__, "pairs_per_family": args.pairs_per_family,
        "cycles": args.cycles, "conditions": model_records,
        "a3": backend.PROTOCOL["a3"], "labels": LABELS,
        "startup_rule": backend.PROTOCOL["startup_rule"]["a3"],
        "source_sha256": {p.name: backend.digest_bytes(p.read_bytes()) for p in sources.iterdir()},
        "pairing": "Same initial state, support drive and split. Closed-loop forces can differ.",
        "acceptance": "Both members must pass primary and half-step QA and refinement. Never resample failures.",
        "training_rows": "analysis_mask & valid_transition, only paired accepted primary traces",
        "event_time": "First geometric contact sample t_i; each row predicts the interval [t_i,t_i+dt].",
        "smoothness_note": "solimp[0]=0 is internally clamped; it is not exactly zero impedance.",
    }
    backend.write_json(root / "protocol.json", protocol)
    backend.write_json(root / "label_thresholds.json", LABELS)
    label_hash = backend.digest_bytes((root / "label_thresholds.json").read_bytes())
    plan = make_plan(args.seed, args.pairs_per_family, args.cycles)
    backend.write_json(root / "candidate_plan_before_simulation.json", {"seed": args.seed, "pairs": plan})
    manifest = {"planned_pairs": len(plan), "smoke": args.smoke,
                "protocol_sha256": backend.digest_bytes((root / "protocol.json").read_bytes()), "pairs": []}
    publish(root, manifest, complete=False)
    print(f"Output: {root}\nPlanned pairs: {len(plan)}; each has two primary and two half-step runs.", flush=True)
    for index, candidate in enumerate(plan, 1):
        members = {}
        for condition in CONDITIONS:
            print(f"[{index}/{len(plan)}] {candidate['id']} — {condition}", flush=True)
            members[condition] = generate_member(root / condition, candidate, args.timestep, label_hash)
        pair = finalize_pair(candidate, members)
        for condition, member in members.items():
            folder = root / condition / "episodes" / candidate["id"]
            backend.write_json(folder / "pair_acceptance.json", {
                "pair_id": candidate["id"], "split": candidate["split"],
                "condition": condition, "accepted": pair["accepted"],
                "member_accepted": member["member_accepted"], "reasons": pair["reasons"],
                "note": "episode.json quality describes this member; this file describes paired inclusion.",
            })
        manifest["pairs"].append(pair)
        publish(root, manifest, complete=False)
        print("  ACCEPTED" if pair["accepted"] else f"  REJECTED: {pair['reasons']}", flush=True)
    publish(root, manifest, complete=True)
    print(f"Accepted {manifest['accepted_pairs']}/{len(plan)} pairs. Training ready: {manifest['training_ready']}")
    print(f"Read {root / 'DATA_REPORT.md'} before training.")


if __name__ == "__main__":
    main()
