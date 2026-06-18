from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_DIR / "learned_policy_weights.json"
REPORT_PATH = PROJECT_DIR / "dataset" / "training_report.json"
FEATURE_NAMES = [
    "progress",
    "raw_error_m",
    "carrying",
    "scanning",
    "disturbance",
    "contact_tracks_norm",
    "sin_phase",
    "cos_phase",
]
OUTPUT_NAMES = ["correction_gain", "policy_confidence", "contact_balance_score"]


def build_dataset(seed: int = 20260618, sample_count: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    progress = rng.uniform(0.0, 1.0, sample_count)
    phase_angle = 2.0 * np.pi * progress
    carrying = ((0.28 <= progress) & (progress <= 0.56)) | ((0.66 <= progress) & (progress <= 0.94))
    scanning = (0.56 <= progress) & (progress <= 0.66)
    disturbance = (0.805 <= progress) & (progress <= 0.875)
    raw_error = rng.uniform(0.008, 0.052, sample_count)
    raw_error += carrying * rng.uniform(0.004, 0.016, sample_count)
    raw_error += disturbance * rng.uniform(0.010, 0.025, sample_count)
    contact_tracks_norm = np.where(carrying, 1.0, np.where(scanning, 0.5, 0.0))

    x = np.column_stack(
        [
            progress,
            raw_error,
            carrying.astype(float),
            scanning.astype(float),
            disturbance.astype(float),
            contact_tracks_norm,
            np.sin(phase_angle),
            np.cos(phase_angle),
        ]
    )
    correction_gain = (
        0.42
        + 2.35 * raw_error
        + 0.18 * carrying
        + 0.09 * scanning
        + 0.12 * disturbance
        + 0.10 * contact_tracks_norm
        + 0.025 * np.sin(phase_angle)
    )
    policy_confidence = (
        0.73
        + 0.16 * progress
        + 0.08 * carrying
        + 0.04 * scanning
        - 0.05 * disturbance
        - 0.55 * raw_error
        + 0.015 * np.cos(phase_angle)
    )
    contact_balance = (
        0.08
        + 0.50 * carrying
        + 0.22 * scanning
        + 0.34 * contact_tracks_norm
        + 0.05 * progress
        - 0.03 * disturbance
    )
    y = np.column_stack(
        [
            np.clip(correction_gain, 0.38, 0.82),
            np.clip(policy_confidence, 0.55, 0.995),
            np.clip(contact_balance, 0.0, 1.0),
        ]
    )
    return x, y


def train() -> dict:
    x, y = build_dataset()
    split = int(0.8 * len(x))
    train_x, val_x = x[:split], x[split:]
    train_y, val_y = y[:split], y[split:]

    design = np.column_stack([train_x, np.ones(len(train_x))])
    params, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    coef = params[:-1]
    intercept = params[-1]
    val_pred = np.column_stack([val_x, np.ones(len(val_x))]) @ params
    val_pred[:, 0] = np.clip(val_pred[:, 0], 0.35, 0.82)
    val_pred[:, 1] = np.clip(val_pred[:, 1], 0.55, 0.995)
    val_pred[:, 2] = np.clip(val_pred[:, 2], 0.0, 1.0)
    mae = np.mean(np.abs(val_pred - val_y), axis=0)

    weights = {
        "policy_type": "learned_linear_residual_grasp_policy",
        "training_method": "least-squares behavioral cloning from randomized perturbation labels",
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
        "project": "Guardian DexTriage Lab",
        "uuid": "e9367728-67e3-4adc-9f3e-fc7a1a364a8d",
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
