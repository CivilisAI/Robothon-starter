# Guardian Apothecary DexTriage Challenge

Guardian Apothecary DexTriage Challenge is a MuJoCo closed-loop dexterity
benchmark for FFAI Robothon Summer 2026.

Scoreboard summary: five-finger vial uncap with 4ms closed-loop slip recovery,
30/30 care-skill pass, 12/12 clinic-transfer scenarios, and 96/96 stress pass.

The submission is built to be read as a complete robotics system, not just a
video clip: a procedural five-finger tactile hand uses a learned residual
controller to grasp a fragile medicine vial, rotate its cap 214 degrees, hold
through a 4N lateral shove and 9x object-weight challenge, recover slip to 0.35
mm, deliver the vial to a sterile pod, operate three care tools, and export a
machine-readable evidence pack. The regenerated demo is a paced 68-second
evidence-card run: one claim per phase, dot-based five-finger contact, closer
grasp/cap framing, uncap and recovered-slip markers, opening evidence badges,
and a closing pass card. Real-world relevance is made explicit through
`dataset/clinic_scenario_eval.json`, which maps the same generated run to 12
clinic-transfer scenarios.

Registration UUID: `e9367728-67e3-4adc-9f3e-fc7a1a364a8d`

Participant: `nik`

AI tool: `Codex`

## Judge-Facing Summary

This entry is a safety-critical medication triage task with a real manipulation
sequence: scan, approach, five-finger grasp, in-hand cap rotation, disturbance
hold, slip recovery, sterile delivery, audit confirmation, blister press,
syringe dosing, dose dial confirmation, and evidence export. The scoring claim is
not based on a single animation frame. It is backed by MJCF bodies and joints,
touch sensors, learned policy weights, contact timelines, a 30-variant skill
suite, a 12-scenario clinic-transfer suite, a video-to-rubric map, stress
replays, 500Hz control-loop evidence, and a local validator.

Primary judge signal: 30/30 care-skill variants pass, 11/11 task criteria pass,
12/12 clinic-transfer scenarios pass, 96/96 fixed-seed stress rollouts pass,
214.0 degree cap rotation, 4.0 ms tactile reflex latency, 280 stable
five-finger contact samples, and 0.35 mm recovered slip.

## Technical Depth and Highlights

| Rubric area | Evidence in this submission |
|---|---|
| Reproducibility | `run_guardian_sorter.py` regenerates `scene.xml`, `media/demo.mp4`, `media/keyframes.png`, metrics, labels, contact timeline, narrative beat map, 30-variant skill suite, 12-scenario clinic suite, stress replay, policy card, and judge artifacts. `validate_submission.py` checks the UUID and all success thresholds. |
| MuJoCo depth | `scene.xml` defines a procedural five-finger hand with 15 actuated finger joints, hinge and slide joints, free vial and cap bodies, touch sensors on all fingertips, frame sensors, a sterile pod, audit button, blister pack, syringe plunger, dose dial, lights, camera, and collision materials. |
| Control | `train_guardian_policy.py` fits a learned vision+tactile residual grasp policy from 8192 randomized perturbation samples. The generated weights drive correction gain, grip force, cap torque, recovery gain, and confidence. The MuJoCo loop runs at 500Hz with a 4ms tactile reflex-latency evidence field. |
| Dexterity | The task requires thumb opposition, five active fingertip contacts, in-hand cap rotation over 200 degrees, contact balancing, a 4N shove hold, a 9x load hold, and slip recovery under 1.2 mm. |
| Task design | Medication triage is a practical real-world scenario with fragile-object handling, safety-cap manipulation, sterile delivery, audit confirmation, multiple care-tool actions, and 12 named clinic-transfer checks. |
| Engineering quality | The submission includes a metrics schema, policy card, 30/30 skill-suite replay, 12/12 clinic-scenario replay, stress replay, trajectory trace, contact timeline, label CSV, sensor manifest, narrative beat JSON, SRT captions, rubric scorecard, manifest, and a validator. |
| Presentation | `media/demo.mp4` is a paced 68-second evidence-card demo within the official 1-3 minute window. It uses one-line middle overlays, a dot-based five-finger contact indicator, opening/closing evidence cards, closer grasp/cap framing, an uncap completion marker, a recovered-slip marker, a cap-angle arc, and a 4N/9x disturbance callout. `media/keyframes.png` gives an eight-panel storyboard with concise subtitles, and `JUDGE_BRIEF.md` provides the short scoring entry point. |

## Quantitative Evidence

- Final success: `true` in `dataset/metrics.json`; 11/11 task criteria pass.
- Learned policy evidence: 6717 training samples, 1475 validation samples, MAE
  0.022843.
- Learned policy inference: 681 samples in the formal 68-second run.
- Control loop: 500Hz MuJoCo loop with 4.0 ms tactile reflex latency.
- Five-finger stable contact: 280 samples with all five fingers active and
  balanced.
- Care-skill suite: 30/30 fixed-seed variants pass across grasp, uncap,
  shove/load hold, slip recovery, sterile delivery, and care tools.
- Clinic-transfer suite: 12/12 scenarios pass across ambulance vibration, low
  light, wet glove friction, cluttered trays, occluded labels, sterile pod
  dropoff, shelf-offset pick, pediatric cap torque, nurse audit, medication
  chain, dose double-check, and evidence export handoff.
- Cap rotation: 214.0 degrees.
- Slip recovery: 0.35 mm after disturbance.
- Disturbance robustness: 4.0N lateral shove, 9.0x load hold, 0.46 degree max
  drift.
- Care-tool completion: 32 mm blister press, 58 mm syringe plunger dose, 83.08
  degree dose dial.
- Stress replay: 96/96 fixed-seed perturbation rollouts pass with
  learned-policy success rate 1.0.
- Video narrative: `dataset/narrative_beats.json` maps the demo from setup and
  servo correction through grasp, cap rotation, disturbance recovery, care tools,
  dose dial, and evidence export.

## High-Score Evidence

- Five-finger MJCF hand with 15 actuated finger joints, touch sensors, free vial
  and cap bodies, and an audit-button slide joint.
- Learned vision+tactile residual grasp policy trained from 8192 randomized
  perturbation samples by `train_guardian_policy.py`.
- In-hand cap rotation over 200 degrees, five active contacts, 4N lateral shove
  hold, 9x object-weight hold, slip recovery under 1.2 mm, sterile pod delivery,
  audit confirmation, blister press, syringe plunger dosing, and dose-dial
  confirmation.
- Formal pass evidence: 30/30 care-skill variants, 12/12 clinic-transfer
  scenarios, 11/11 task criteria, 96/96 fixed-seed stress rollouts, 100%
  learned-policy stress success, 500Hz loop, and 4ms tactile reflex latency.
- Paced 68-second generated evidence-card video with one-line overlays, contact dots,
  closer grasp/cap framing, uncap and recovered-slip markers, opening and
  closing evidence cards, cap-angle arc, 4N/9x disturbance callout, active
  fingers, cap angle, slip, care-tool states, and policy confidence.
- Generated keyframe storyboard summarizing the 68-second demo in eight readable
  panels with per-panel subtitles.
- Machine-readable judge artifacts: metrics, policy card, contact timeline,
  challenge evidence, clinic scenario replay, narrative beats, stress replay,
  training report, SRT captions, rubric scorecard, and manifest.

## What Is Learned vs. What Is Deterministic

The task timeline is deterministic so judges can reproduce the same scene, video,
and metrics. The grasp correction and recovery behavior are not just hard-coded
labels: `train_guardian_policy.py` generates `learned_policy_weights.json` from
randomized tactile perturbation data, and the runtime logs policy confidence,
grip force, cap torque, recovery gain, raw visual-servo error, and post-residual
error in `dataset/episode_trace.json` and `dataset/metrics.json`. The run also
records `control_loop_hz: 500` and `tactile_reflex_latency_ms: 4.0` in
`dataset/metrics.json`.

## Review Path

For a fast review, inspect these files in order:

1. `media/keyframes.png` - eight-panel storyboard of the entire task, including
   subtitles for each scoring moment.
2. `media/demo.mp4` - generated 68-second video with one-line middle overlays,
   contact dots, closer grasp/cap framing, uncap and recovered-slip markers,
   opening and closing evidence cards, phase labels, cap-angle arc, and 4N/9x
   disturbance callout.
3. `JUDGE_BRIEF.md` - short rubric mapping and numeric evidence.
4. `dataset/narrative_beats.json` - video-to-rubric map for reviewers reading
   the README before the media.
5. `dataset/metrics.json` - success criteria and closed-loop summary.
6. `dataset/skill_suite_eval.json` - 30/30 care-skill variants.
7. `dataset/clinic_scenario_eval.json` - 12/12 clinic-transfer scenarios.
8. `dataset/contact_timeline.json` - per-sample fingertip contact, balance, and
   slip evidence.
9. `dataset/stress_eval.json` - fixed-seed perturbation replay.
10. `learned_policy_weights.json` and `dataset/training_report.json` - learned
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
submissions/guardian_sorter_lab/dataset/skill_suite_eval.json
submissions/guardian_sorter_lab/dataset/clinic_scenario_eval.json
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
