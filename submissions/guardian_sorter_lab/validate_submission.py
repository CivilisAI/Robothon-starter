from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
UUID = "e9367728-67e3-4adc-9f3e-fc7a1a364a8d"
REQUIRED_FILES = [
    "registration.json",
    "task_config.json",
    "scene.xml",
    "run_guardian_sorter.py",
    "train_guardian_policy.py",
    "learned_policy_weights.json",
    "README.md",
    "JUDGE_BRIEF.md",
    "rubric_scorecard.json",
    "submission_manifest.json",
    "media/demo.mp4",
    "dataset/episode_trace.json",
    "dataset/labels.csv",
    "dataset/metrics.json",
    "dataset/sensor_manifest.json",
    "dataset/contact_timeline.json",
    "dataset/stress_eval.json",
    "dataset/training_report.json",
    "dataset/policy_card.json",
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
    scorecard = load_json("rubric_scorecard.json")
    manifest = load_json("submission_manifest.json")
    policy_card = load_json("dataset/policy_card.json")
    weights = load_json("learned_policy_weights.json")
    training_report = load_json("dataset/training_report.json")
    stress_eval = load_json("dataset/stress_eval.json")
    contact_timeline = load_json("dataset/contact_timeline.json")

    uuid_values = [
        registration.get("uuid"),
        metrics.get("uuid"),
        scorecard.get("registration_uuid"),
        manifest.get("uuid"),
        policy_card.get("uuid"),
        training_report.get("uuid"),
    ]
    if uuid_values != [UUID] * len(uuid_values):
        raise SystemExit(f"UUID mismatch: {uuid_values}")

    if not metrics.get("success"):
        raise SystemExit("Metrics did not report success=true")
    if not metrics.get("criteria", {}).get("disturbance_recovery_observed"):
        raise SystemExit("Disturbance recovery evidence is missing")
    if metrics.get("closed_loop_summary", {}).get("residual_corrections", 0) < 50:
        raise SystemExit("Too few residual correction samples")
    if metrics.get("closed_loop_summary", {}).get("policy_training_samples", 0) < 3000:
        raise SystemExit("Learned policy training evidence is missing")
    if weights.get("policy_type") != metrics.get("closed_loop_summary", {}).get("policy_type"):
        raise SystemExit("Policy type mismatch between weights and metrics")
    if stress_eval.get("residual_policy_success_rate", 0.0) < 0.95:
        raise SystemExit("Stress replay residual success rate below target")
    if not any(row.get("stable_hold") for row in contact_timeline):
        raise SystemExit("Contact timeline has no stable hold samples")

    video_path = PROJECT_DIR / "media" / "demo.mp4"
    if video_path.stat().st_size < 1_000_000:
        raise SystemExit("Demo video is unexpectedly small")

    print(
        json.dumps(
            {
                "ok": True,
                "uuid": UUID,
                "video_bytes": video_path.stat().st_size,
                "residual_corrections": metrics["closed_loop_summary"]["residual_corrections"],
                "policy_type": metrics["closed_loop_summary"]["policy_type"],
                "policy_training_samples": metrics["closed_loop_summary"]["policy_training_samples"],
                "stress_success": stress_eval["residual_policy_success_rate"],
                "stable_contact_samples": metrics["closed_loop_summary"]["stable_contact_samples"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
