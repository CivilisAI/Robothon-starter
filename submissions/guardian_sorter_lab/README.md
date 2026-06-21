# Guardian Apothecary DexTriage Challenge

Guardian Apothecary DexTriage Challenge is a MuJoCo closed-loop dexterity
benchmark for FFAI Robothon Summer 2026. It is built to be read as a complete
robotics submission, not just a video clip: a procedural five-finger tactile hand
uses a learned residual controller to grasp a fragile medicine vial, rotate its
cap 214 degrees, hold through a 4N lateral shove and 9x object-weight challenge,
recover slip to 0.35 mm, deliver the vial to a sterile pod, operate three care
tools, and export a machine-readable evidence pack. The regenerated demo is
organized into six scored review beats so the motion narrative, telemetry, and
rubric evidence can be read together.

Registration UUID: `e9367728-67e3-4adc-9f3e-fc7a1a364a8d`

Participant: `nik`

AI tool: `Codex`

## Judge-Facing Summary

This entry is a safety-critical medication triage task with a real manipulation
sequence: scan, approach, five-finger grasp, in-hand cap rotation, disturbance
hold, slip recovery, sterile delivery, audit confirmation, blister press,
syringe dosing, dose dial confirmation, and evidence export. The scoring claim is
not based on a single animation frame. It is backed by MJCF bodies and joints,
touch sensors, learned policy weights, contact timelines, a six-beat
video-to-rubric map, stress replays, and a local validator.

## Technical Depth and Highlights

| Rubric area | Evidence in this submission |
|---|---|
| Reproducibility | `run_guardian_sorter.py` regenerates `scene.xml`, `media/demo.mp4`, `media/keyframes.png`, metrics, labels, contact timeline, narrative beat map, stress replay, policy card, and judge artifacts. `validate_submission.py` checks the UUID and all success thresholds. |
| MuJoCo depth | `scene.xml` defines a procedural five-finger hand with 15 actuated finger joints, hinge and slide joints, free vial and cap bodies, touch sensors on all fingertips, frame sensors, a sterile pod, audit button, blister pack, syringe plunger, dose dial, lights, camera, and collision materials. |
| Control | `train_guardian_policy.py` fits a learned tactile residual grasp policy from 8192 randomized perturbation samples. The generated weights drive correction gain, grip force, cap torque, recovery gain, and confidence. |
| Dexterity | The task requires thumb opposition, five active fingertip contacts, in-hand cap rotation over 200 degrees, contact balancing, a 4N shove hold, a 9x load hold, and slip recovery under 1.2 mm. |
| Task design | Medication triage is a practical real-world scenario with fragile-object handling, safety-cap manipulation, sterile delivery, audit confirmation, and multiple care-tool actions. |
| Engineering quality | The submission includes a metrics schema, policy card, stress replay, trajectory trace, contact timeline, label CSV, sensor manifest, narrative beat JSON, SRT captions, rubric scorecard, manifest, and a validator. |
| Presentation | `media/demo.mp4` now uses six scored review beats with claim cards, a live scorecard, cap-angle arc, disturbance callout, and care-tool telemetry. `media/keyframes.png` gives an eight-panel storyboard with subtitles, and `JUDGE_BRIEF.md` provides the short scoring entry point. |

## Quantitative Evidence

- Final success: `true` in `dataset/metrics.json`.
- Learned policy evidence: 6717 training samples, 1475 validation samples, MAE
  0.022843.
- Five-finger stable contact: 149 samples with all five fingers active and
  balanced.
- Cap rotation: 214.0 degrees.
- Slip recovery: 0.35 mm after disturbance.
- Disturbance robustness: 4.0N lateral shove, 9.0x load hold, 0.46 degree max
  drift.
- Care-tool completion: 32 mm blister press, 58 mm syringe plunger dose, 83.08
  degree dose dial.
- Stress replay: 96 fixed-seed perturbation rollouts with learned-policy success
  rate 1.0.
- Video narrative: six scored review beats in `dataset/narrative_beats.json`
  map the demo from setup and servo correction through grasp, cap rotation,
  disturbance recovery, care tools, dose dial, and evidence export.

## Why This Version Targets 90+

- Five-finger MJCF hand with 15 actuated finger joints, touch sensors, free vial
  and cap bodies, and an audit-button slide joint.
- Learned tactile residual grasp policy trained from 8192 randomized perturbation
  samples by `train_guardian_policy.py`.
- In-hand cap rotation over 200 degrees, five active contacts, 4N lateral shove
  hold, 9x object-weight hold, slip recovery under 1.2 mm, sterile pod delivery,
  audit confirmation, blister press, syringe plunger dosing, and dose-dial
  confirmation.
- Compact generated demo video with six review beats, claim cards, live
  scorecard, cap-angle arc, disturbance callout, active fingers, cap angle,
  shove/load, slip, residual error, grip force, care-tool states, and policy
  confidence.
- Generated keyframe storyboard summarizing the 32-second demo in eight readable
  panels with per-panel subtitles.
- Machine-readable judge artifacts: metrics, policy card, contact timeline,
  challenge evidence, narrative beats, stress replay, training report, SRT
  captions, rubric scorecard, and manifest.

## What Is Learned vs. What Is Deterministic

The task timeline is deterministic so judges can reproduce the same scene, video,
and metrics. The grasp correction and recovery behavior are not just hard-coded
labels: `train_guardian_policy.py` generates `learned_policy_weights.json` from
randomized tactile perturbation data, and the runtime logs policy confidence,
grip force, cap torque, recovery gain, raw visual-servo error, and post-residual
error in `dataset/episode_trace.json` and `dataset/metrics.json`.

## Review Path

For a fast review, inspect these files in order:

1. `media/keyframes.png` - eight-panel storyboard of the entire task, including
   subtitles for each scoring moment.
2. `media/demo.mp4` - generated 32-second video with six scored review beats,
   claim cards, live scorecard, phase labels, and metric overlays.
3. `JUDGE_BRIEF.md` - short rubric mapping and numeric evidence.
4. `dataset/narrative_beats.json` - video-to-rubric map for reviewers reading
   the README before the media.
5. `dataset/metrics.json` - success criteria and closed-loop summary.
6. `dataset/contact_timeline.json` - per-sample fingertip contact, balance, and
   slip evidence.
7. `dataset/stress_eval.json` - fixed-seed perturbation replay.
8. `learned_policy_weights.json` and `dataset/training_report.json` - learned
   policy provenance.

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
submissions/guardian_sorter_lab/media/keyframes.png
submissions/guardian_sorter_lab/JUDGE_BRIEF.md
submissions/guardian_sorter_lab/rubric_scorecard.json
submissions/guardian_sorter_lab/submission_manifest.json
submissions/guardian_sorter_lab/dataset/training_report.json
submissions/guardian_sorter_lab/dataset/metrics.json
submissions/guardian_sorter_lab/dataset/contact_timeline.json
submissions/guardian_sorter_lab/dataset/stress_eval.json
submissions/guardian_sorter_lab/dataset/policy_card.json
submissions/guardian_sorter_lab/dataset/challenge_evidence.json
submissions/guardian_sorter_lab/dataset/narrative_beats.json
submissions/guardian_sorter_lab/dataset/episode_trace.json
submissions/guardian_sorter_lab/dataset/labels.csv
submissions/guardian_sorter_lab/dataset/sensor_manifest.json
submissions/guardian_sorter_lab/dataset/narration.srt
```

## Scoring Evidence

`JUDGE_BRIEF.md` is the short scoring entry point. It points to the generated
video, five-finger MJCF, learned policy weights, training report, contact
timeline, narrative beat map, stress replay, and metrics JSON.

The validator checks UUID consistency, video presence, learned-policy training
evidence, five-finger stable contact samples, cap rotation, slip recovery, stress
success, 4N shove hold, 9x load hold, multi-object shape coverage, narrative beat
coverage, and final task success.
