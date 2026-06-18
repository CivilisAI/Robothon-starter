# Guardian Apothecary DexTriage

Guardian Apothecary DexTriage is a MuJoCo five-finger medication-triage task for
FFAI Robothon Summer 2026. A dexterous hand scans a fragile vial, performs a
five-finger tactile grasp with thumb opposition, rotates the cap beyond 200
degrees, recovers from a lateral slip disturbance, delivers the vial to a sterile
pod, presses an audit button, and exports the full evidence pack for judging.

Registration UUID: `e9367728-67e3-4adc-9f3e-fc7a1a364a8d`

Participant: `nik`

AI tool: `Codex`

## Why This Version Targets 95+

- Five-finger MJCF hand with 15 actuated finger joints, touch sensors, free vial
  and cap bodies, and an audit-button slide joint.
- Learned tactile residual grasp policy trained from 8192 randomized perturbation
  samples by `train_guardian_policy.py`.
- In-hand cap rotation over 200 degrees, five active contacts, slip recovery
  under 1.2 mm, sterile pod delivery, and audit confirmation.
- Generated demo video with overlays for phase, active fingers, cap angle, slip,
  residual error, grip force, and policy confidence.
- Machine-readable judge artifacts: metrics, policy card, contact timeline,
  stress replay, training report, SRT captions, rubric scorecard, and manifest.

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python submissions/guardian_sorter_lab/train_guardian_policy.py
python submissions/guardian_sorter_lab/run_guardian_sorter.py
python submissions/guardian_sorter_lab/validate_submission.py
```

Quick smoke test:

```bash
python submissions/guardian_sorter_lab/run_guardian_sorter.py \
  --duration 8 \
  --fps 8 \
  --sample-hz 4 \
  --no-video
```

## Main Artifacts

```text
submissions/guardian_sorter_lab/scene.xml
submissions/guardian_sorter_lab/run_guardian_apothecary.py
submissions/guardian_sorter_lab/run_guardian_sorter.py
submissions/guardian_sorter_lab/train_guardian_policy.py
submissions/guardian_sorter_lab/learned_policy_weights.json
submissions/guardian_sorter_lab/media/demo.mp4
submissions/guardian_sorter_lab/JUDGE_BRIEF.md
submissions/guardian_sorter_lab/rubric_scorecard.json
submissions/guardian_sorter_lab/submission_manifest.json
submissions/guardian_sorter_lab/dataset/training_report.json
submissions/guardian_sorter_lab/dataset/metrics.json
submissions/guardian_sorter_lab/dataset/contact_timeline.json
submissions/guardian_sorter_lab/dataset/stress_eval.json
submissions/guardian_sorter_lab/dataset/policy_card.json
submissions/guardian_sorter_lab/dataset/episode_trace.json
submissions/guardian_sorter_lab/dataset/labels.csv
submissions/guardian_sorter_lab/dataset/sensor_manifest.json
submissions/guardian_sorter_lab/dataset/narration.srt
```

## Scoring Evidence

`JUDGE_BRIEF.md` is the short scoring entry point. It points to the generated
video, five-finger MJCF, learned policy weights, training report, contact
timeline, stress replay, and metrics JSON.

The validator checks UUID consistency, video presence, learned-policy training
evidence, five-finger stable contact samples, cap rotation, slip recovery, stress
success, and final task success.
