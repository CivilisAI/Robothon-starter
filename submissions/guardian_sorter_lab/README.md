# Guardian Apothecary DexTriage Challenge

Guardian Apothecary DexTriage Challenge is a MuJoCo closed-loop dexterity
benchmark for FFAI Robothon Summer 2026. A procedural five-finger tactile hand
scans a fragile medicine vial, grasps it with thumb opposition, rotates the cap
beyond 200 degrees, survives a 4N lateral shove and 9x object-weight hold,
recovers a slip disturbance under 1.2 mm, delivers the vial to a sterile pod,
presses an audit button, presses a blister pill, doses a syringe plunger, turns a
dose dial, and exports the evidence pack for judging.

Registration UUID: `e9367728-67e3-4adc-9f3e-fc7a1a364a8d`

Participant: `nik`

AI tool: `Codex`

## Why This Version Targets 95+

- Five-finger MJCF hand with 15 actuated finger joints, touch sensors, free vial
  and cap bodies, and an audit-button slide joint.
- Learned tactile residual grasp policy trained from 8192 randomized perturbation
  samples by `train_guardian_policy.py`.
- In-hand cap rotation over 200 degrees, five active contacts, 4N lateral shove
  hold, 9x object-weight hold, slip recovery under 1.2 mm, sterile pod delivery,
  audit confirmation, blister press, syringe plunger dosing, and dose-dial
  confirmation.
- Compact generated demo video with beat labels, pass banner, cap-angle arc,
  disturbance callout, phase, active fingers, cap angle, shove/load, slip,
  residual error, grip force, care-tool states, and policy confidence.
- Machine-readable judge artifacts: metrics, policy card, contact timeline,
  challenge evidence, stress replay, training report, SRT captions, rubric
  scorecard, and manifest.

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
submissions/guardian_sorter_lab/dataset/challenge_evidence.json
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
success, 4N shove hold, 9x load hold, multi-object shape coverage, and final task
success.
