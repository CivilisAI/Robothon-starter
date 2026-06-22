from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_DIR / "learned_policy_weights.json"
REPORT_PATH = PROJECT_DIR / "dataset" / "training_report.json"
UUID = "e9367728-67e3-4adc-9f3e-fc7a1a364a8d"
FEATURE_NAMES = [
    "progress",
    "vial_error_m",
    "cap_error_rad",
    "slip_mm",
    "active_fingers_norm",
    "contact_balance",
    "disturbance",
    "sin_phase",
    "cos_phase",
]
OUTPUT_NAMES = [
    "palm_correction_gain",
    "grip_force",
    "cap_torque",
    "recovery_gain",
    "policy_confidence",
]


def build_dataset(seed: int = 20260618, sample_count: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    progress = rng.uniform(0.0, 1.0, sample_count)
    phase = 2.0 * np.pi * progress
    manipulation = ((0.22 <= progress) & (progress <= 0.88)).astype(float)
    cap_turning = ((0.38 <= progress) & (progress <= 0.60)).astype(float)
    disturbance = ((0.58 <= progress) & (progress <= 0.70)).astype(float)
    vial_error = rng.uniform(0.002, 0.055, sample_count) + disturbance * rng.uniform(0.008, 0.030, sample_count)
    cap_error = rng.uniform(0.0, 2.2, sample_count) * cap_turning
    slip_mm = rng.uniform(0.2, 8.0, sample_count) + disturbance * rng.uniform(2.0, 10.0, sample_count)
    active_fingers_norm = np.clip(
        manipulation * rng.normal(0.88, 0.12, sample_count) + (1.0 - manipulation) * rng.normal(0.08, 0.04, sample_count),
        0.0,
        1.0,
    )
    contact_balance = np.clip(
        active_fingers_norm - 0.16 * disturbance + rng.normal(0.0, 0.055, sample_count),
        0.0,
        1.0,
    )

    x = np.column_stack(
        [
            progress,
            vial_error,
            cap_error,
            slip_mm,
            active_fingers_norm,
            contact_balance,
            disturbance,
            np.sin(phase),
            np.cos(phase),
        ]
    )
    palm_gain = 0.44 + 2.25 * vial_error + 0.05 * cap_turning + 0.09 * disturbance + 0.11 * contact_balance
    grip_force = 0.18 + 0.62 * manipulation + 0.18 * active_fingers_norm + 0.025 * slip_mm + 0.07 * disturbance
    cap_torque = 0.10 + 0.34 * cap_turning + 0.18 * np.tanh(cap_error) + 0.08 * contact_balance
    recovery_gain = 0.22 + 0.55 * disturbance + 0.030 * slip_mm + 0.12 * (1.0 - contact_balance)
    confidence = 0.66 + 0.22 * contact_balance + 0.07 * manipulation - 0.006 * slip_mm - 0.04 * disturbance
    y = np.column_stack(
        [
            np.clip(palm_gain, 0.35, 0.88),
            np.clip(grip_force, 0.0, 1.0),
            np.clip(cap_torque, 0.0, 0.82),
            np.clip(recovery_gain, 0.0, 1.0),
            np.clip(confidence, 0.45, 0.995),
        ]
    )
    return x, y


def train() -> dict:
    x, y = build_dataset()
    split = int(0.82 * len(x))
    train_x, val_x = x[:split], x[split:]
    train_y, val_y = y[:split], y[split:]
    design = np.column_stack([train_x, np.ones(len(train_x))])
    params, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    coef = params[:-1]
    intercept = params[-1]
    val_pred = np.column_stack([val_x, np.ones(len(val_x))]) @ params
    val_pred[:, 0] = np.clip(val_pred[:, 0], 0.35, 0.88)
    val_pred[:, 1] = np.clip(val_pred[:, 1], 0.0, 1.0)
    val_pred[:, 2] = np.clip(val_pred[:, 2], 0.0, 0.82)
    val_pred[:, 3] = np.clip(val_pred[:, 3], 0.0, 1.0)
    val_pred[:, 4] = np.clip(val_pred[:, 4], 0.45, 0.995)
    mae = np.mean(np.abs(val_pred - val_y), axis=0)
    weights = {
        "policy_type": "learned_tactile_residual_grasp_policy",
        "training_method": "least-squares behavioral cloning from randomized tactile grasp perturbation labels",
        "training_samples": int(len(train_x)),
        "validation_samples": int(len(val_x)),
        "feature_names": FEATURE_NAMES,
        "output_names": OUTPUT_NAMES,
        "coef": np.round(coef, 8).tolist(),
        "intercept": np.round(intercept, 8).tolist(),
        "validation": {
            "mean_absolute_error": round(float(np.mean(mae)), 6),
            "per_output_mae": {name: round(float(value), 6) for name, value in zip(OUTPUT_NAMES, mae, strict=True)},
        },
    }
    report = {
        "project": "Guardian Apothecary DexTriage Challenge",
        "uuid": UUID,
        "policy_type": weights["policy_type"],
        "training_method": weights["training_method"],
        "training_samples": weights["training_samples"],
        "validation_samples": weights["validation_samples"],
        "validation": weights["validation"],
        "feature_names": FEATURE_NAMES,
        "output_names": OUTPUT_NAMES,
    }
    WEIGHTS_PATH.write_text(json.dumps(weights, indent=2), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    print(json.dumps(train(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
