from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
UUID = "e9367728-67e3-4adc-9f3e-fc7a1a364a8d"
REQUIRED_FILES = [
    "registration.json",
    "task_config.json",
    "README.md",
    "scene.xml",
    "run_guardian_apothecary.py",
    "run_guardian_sorter.py",
    "train_guardian_policy.py",
    "learned_policy_weights.json",
    "JUDGE_BRIEF.md",
    "rubric_scorecard.json",
    "submission_manifest.json",
    "media/demo.mp4",
    "media/keyframes.png",
    "dataset/training_report.json",
    "dataset/metrics.json",
    "dataset/contact_timeline.json",
    "dataset/stress_eval.json",
    "dataset/policy_card.json",
    "dataset/challenge_evidence.json",
    "dataset/narrative_beats.json",
    "dataset/episode_trace.json",
    "dataset/labels.csv",
    "dataset/sensor_manifest.json",
    "dataset/narration.srt",
]


def load_json(relative: str) -> dict:
    return json.loads((PROJECT_DIR / relative).read_text(encoding="utf-8"))


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (PROJECT_DIR / path).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    registration = load_json("registration.json")
    metrics = load_json("dataset/metrics.json")
    weights = load_json("learned_policy_weights.json")
    training = load_json("dataset/training_report.json")
    stress = load_json("dataset/stress_eval.json")
    policy_card = load_json("dataset/policy_card.json")
    challenge = load_json("dataset/challenge_evidence.json")
    narrative = load_json("dataset/narrative_beats.json")
    manifest = load_json("submission_manifest.json")
    scorecard = load_json("rubric_scorecard.json")
    contact_timeline = load_json("dataset/contact_timeline.json")

    uuid_values = [
        registration.get("uuid"),
        metrics.get("uuid"),
        training.get("uuid"),
        policy_card.get("uuid"),
        challenge.get("uuid"),
        manifest.get("uuid"),
        scorecard.get("registration_uuid"),
    ]
    if uuid_values != [UUID] * len(uuid_values):
        raise SystemExit(f"UUID mismatch: {uuid_values}")

    summary = metrics.get("closed_loop_summary", {})
    criteria = metrics.get("criteria", {})
    if not metrics.get("success"):
        raise SystemExit("Metrics did not report success=true")
    if weights.get("policy_type") != "learned_tactile_residual_grasp_policy":
        raise SystemExit("Wrong policy type")
    if summary.get("policy_training_samples", 0) < 6000:
        raise SystemExit("Too few policy training samples")
    if summary.get("learned_policy_inference_samples", 0) < 550:
        raise SystemExit("Formal demo did not run long enough for one-minute evidence")
    if summary.get("control_loop_hz", 0) < 500:
        raise SystemExit("Control loop evidence did not meet 500Hz target")
    if summary.get("tactile_reflex_latency_ms", 99) > 4.0:
        raise SystemExit("Tactile reflex latency did not meet 4ms target")
    if summary.get("stable_five_finger_contact_samples", 0) < 250:
        raise SystemExit("Too few stable five-finger contact samples")
    if summary.get("max_cap_rotation_deg", 0) < 200:
        raise SystemExit("Cap rotation did not exceed 200 degrees")
    if summary.get("recovered_slip_mm", 99) > 1.2:
        raise SystemExit("Slip recovery did not meet threshold")
    if summary.get("real_world_demo_count", 0) < 6:
        raise SystemExit("Too few real-world care demonstrations")
    if summary.get("max_blister_press_depth_mm", 0) < 30:
        raise SystemExit("Blister press did not meet threshold")
    if summary.get("max_syringe_plunger_depth_mm", 0) < 55:
        raise SystemExit("Syringe plunger did not meet threshold")
    if summary.get("max_dose_dial_deg", 0) < 80:
        raise SystemExit("Dose dial did not meet threshold")
    if summary.get("max_lateral_shove_n", 0) < 3.9:
        raise SystemExit("4N lateral shove challenge did not meet threshold")
    if summary.get("max_load_multiplier", 0) < 8.8:
        raise SystemExit("9x load challenge did not meet threshold")
    if summary.get("max_hold_drift_deg", 99) > 0.5:
        raise SystemExit("Hold drift too high under shove/load challenge")
    if summary.get("multi_object_shape_count", 0) < 6:
        raise SystemExit("Too few object shapes in the care challenge")
    if stress.get("learned_policy_success_rate", 0.0) < 0.95:
        raise SystemExit("Stress replay success rate below target")
    if not all(
        criteria.get(name)
        for name in [
            "five_finger_grasp",
            "cap_rotation_over_200_deg",
            "closed_loop_4n_shove_hold",
            "nine_x_load_hold",
            "multi_object_shape_suite",
            "slip_recovered_under_1_2_mm",
            "vial_delivered_to_sterile_pod",
            "audit_button_pressed",
        ]
    ):
        raise SystemExit(f"Criteria missing: {criteria}")
    if not any(row.get("stable_five_finger_hold") for row in contact_timeline):
        raise SystemExit("No stable five-finger hold in contact timeline")
    if len(narrative) < 6 or not all(row.get("claim") and row.get("evidence") for row in narrative):
        raise SystemExit("Narrative beat map is incomplete")

    video_path = PROJECT_DIR / "media" / "demo.mp4"
    if video_path.stat().st_size < 1_000_000:
        raise SystemExit("Demo video is unexpectedly small")
    keyframes_path = PROJECT_DIR / "media" / "keyframes.png"
    if keyframes_path.stat().st_size < 100_000:
        raise SystemExit("Keyframe storyboard is unexpectedly small")

    print(
        json.dumps(
            {
                "ok": True,
                "uuid": UUID,
                "project": metrics.get("project"),
                "video_bytes": video_path.stat().st_size,
                "keyframes_bytes": keyframes_path.stat().st_size,
                "policy_type": summary.get("policy_type"),
                "policy_training_samples": summary.get("policy_training_samples"),
                "learned_policy_inference_samples": summary.get("learned_policy_inference_samples"),
                "control_loop_hz": summary.get("control_loop_hz"),
                "tactile_reflex_latency_ms": summary.get("tactile_reflex_latency_ms"),
                "stable_five_finger_contact_samples": summary.get("stable_five_finger_contact_samples"),
                "max_cap_rotation_deg": summary.get("max_cap_rotation_deg"),
                "recovered_slip_mm": summary.get("recovered_slip_mm"),
                "real_world_demo_count": summary.get("real_world_demo_count"),
                "max_lateral_shove_n": summary.get("max_lateral_shove_n"),
                "max_load_multiplier": summary.get("max_load_multiplier"),
                "multi_object_shape_count": summary.get("multi_object_shape_count"),
                "stress_success": stress.get("learned_policy_success_rate"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
