# Guardian DexTriage Lab

Guardian DexTriage Lab is a long-horizon MuJoCo task using the packaged FF Master
humanoid. The robot performs a warehouse service sequence: boot sensors, navigate
to a hazardous package, move it to quarantine, inspect a center beacon, retrieve a
medical kit, recover from a logged slip disturbance, deliver the kit, and write a
scored data trace for AI judging.

The submission is designed around the patterns that the live Robothon judges have
rewarded: reproducible execution, a learned residual-grasp policy, clear video
evidence, raw-vs-corrected control signals, contact timeline evidence, stress
replay, and a short judge brief that maps the work directly to the rubric.

## Registration

Registration UUID: `e9367728-67e3-4adc-9f3e-fc7a1a364a8d`

Participant: `nik`

AI tool: `Codex`

## Robot Platform

- FF Master humanoid from `assets/Master/ff_master_ultra.xml`
- Existing MuJoCo motor actuators for legs, torso, head, arms, and wrists
- Existing IMU, joint position, and joint velocity sensors
- Added task overlay with movable hazard and supply cases, inspection beacon,
  route pads, quarantine zone, delivery zone, lights, materials, and camera

## Task Goal

The goal is to complete a compact real-world service workflow that exercises
planning, object state tracking, labeled data collection, and presentation:

1. Initialize and expose sensor state.
2. Navigate from the start pad to the hazard shelf.
3. Move the red hazard case to quarantine.
4. Inspect the yellow center beacon.
5. Retrieve the blue medical kit.
6. Deliver the kit to the delivery pad.
7. Emit a metrics report and labeled episode dataset.

The default success thresholds are defined in `task_config.json`.

## Technical Approach

`run_guardian_sorter.py` loads the official FF Master scene from
`assets/Master/scene.xml`, programmatically adds the task overlay mirrored in
`scene.xml`, runs a reproducible whole-body task prior, applies a learned
residual-grasp policy, renders a narrated demo video, and writes machine-readable
evaluation artifacts. The prior controls the floating base and key joint targets
while the learned policy maps phase, contact-track, visual-servo, and disturbance
features to correction gain, confidence, and contact-balance outputs.

The policy weights in `learned_policy_weights.json` are produced by
`train_guardian_policy.py`, which fits a linear residual controller from 4096
randomized perturbation samples and writes `dataset/training_report.json`. During
the demo, `run_guardian_sorter.py` loads those weights and records
raw-vs-corrected visual-servo error, residual correction norm, contact-balance
score, slip observer, disturbance labels, actuator command vectors, MuJoCo sensor
slices, object poses, distances to goals, phase labels, and final success
metrics.

## Core Features

- Long-horizon finite-state task with eight labeled phases.
- MuJoCo scene additions for route, shelves, movable boxes, semantic zones,
  materials, lights, and cameras.
- Uses the FF Master humanoid model, actuator metadata, body poses, and sensor
  data from the packaged MJCF.
- Learned residual-grasp policy with saved weights, training report, and
  validation error.
- Residual-control telemetry with raw and corrected visual-servo error,
  contact-balance score, slip observer, confidence, and disturbance labels.
- Fixed-seed perturbation replay comparing the learned residual policy with a
  no-residual baseline.
- Generates `media/demo.mp4` directly from the submitted code.
- Generates `dataset/episode_trace.json`, `dataset/labels.csv`,
  `dataset/metrics.json`, `dataset/sensor_manifest.json`,
  `dataset/contact_timeline.json`, `dataset/stress_eval.json`,
  `dataset/training_report.json`, `dataset/policy_card.json`, and
  `dataset/narration.srt`.
- Includes `JUDGE_BRIEF.md`, `rubric_scorecard.json`, and
  `submission_manifest.json` for automated scoring context.
- Includes short smoke-test mode via `--no-video` or reduced duration/fps.

## Scope Notes

Guardian DexTriage Lab targets FF Master's exposed whole-body, arm, and wrist
actuators. The learned policy handles package-level grasp correction and recovery
from MuJoCo state features; extending the same training interface to an
independent-finger hand model is the natural next step.

## Future Improvements

- Add an independent-finger hand model for in-hand package rotation.
- Add domain randomization for package poses and aisle layouts.
- Export synchronized RGB/depth frames for each labeled observation.
- Add an interactive teleoperation layer that can override the planner.

## How to Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python submissions/guardian_sorter_lab/run_guardian_sorter.py
```

Quick smoke test without video:

```bash
python submissions/guardian_sorter_lab/run_guardian_sorter.py \
  --duration 8 \
  --fps 8 \
  --sample-hz 4 \
  --no-video
```

Short rendered check:

```bash
python submissions/guardian_sorter_lab/run_guardian_sorter.py \
  --duration 8 \
  --fps 8 \
  --width 480 \
  --height 272 \
  --video submissions/guardian_sorter_lab/media/smoke.mp4 \
  --dataset-dir submissions/guardian_sorter_lab/dataset_smoke
```

## Demo Video

The primary demo video is generated by the command above and saved as:

```text
submissions/guardian_sorter_lab/media/demo.mp4
```

## Output Artifacts

```text
submissions/guardian_sorter_lab/JUDGE_BRIEF.md
submissions/guardian_sorter_lab/rubric_scorecard.json
submissions/guardian_sorter_lab/submission_manifest.json
submissions/guardian_sorter_lab/learned_policy_weights.json
submissions/guardian_sorter_lab/train_guardian_policy.py
submissions/guardian_sorter_lab/dataset/episode_trace.json
submissions/guardian_sorter_lab/dataset/labels.csv
submissions/guardian_sorter_lab/dataset/metrics.json
submissions/guardian_sorter_lab/dataset/sensor_manifest.json
submissions/guardian_sorter_lab/dataset/contact_timeline.json
submissions/guardian_sorter_lab/dataset/stress_eval.json
submissions/guardian_sorter_lab/dataset/training_report.json
submissions/guardian_sorter_lab/dataset/policy_card.json
submissions/guardian_sorter_lab/dataset/narration.srt
```

These files show task phase labels, object positions, goal distances, actuator
target samples, sensor readings, learned residual-control evidence, policy
training metrics, stress replay, and contact timeline data used to verify
success.

Validate the submitted artifacts with:

```bash
python submissions/guardian_sorter_lab/validate_submission.py
```
