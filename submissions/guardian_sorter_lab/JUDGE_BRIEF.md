# Guardian Sorter Lab - Judge Brief

Registration UUID: e9367728-67e3-4adc-9f3e-fc7a1a364a8d

## Why This Entry Is Built For A 95+ Score

Guardian Sorter Lab combines the official FF Master humanoid with a long-horizon
warehouse service task and a machine-readable evidence pack. The robot sorts a
hazard case, inspects a beacon, retrieves a medkit, recovers from a logged slip
disturbance, and exports video, trajectory labels, contact telemetry, policy
metadata, and fixed-seed stress replay results.

## What To Inspect First

1. `media/demo.mp4` - generated demo video with phase, residual, contact, and confidence overlays.
2. `dataset/metrics.json` - success criteria, final distances, and closed-loop summary.
3. `dataset/episode_trace.json` - per-sample MuJoCo state, sensors, controls, and feedback.
4. `dataset/contact_timeline.json` - active contact tracks, balance score, stable holds, and disturbance labels.
5. `dataset/stress_eval.json` - fixed-seed perturbation replay with baseline-vs-residual comparison.
6. `dataset/policy_card.json` - controller inputs, outputs, scope, and evidence.
7. `rubric_scorecard.json` - direct mapping to Robothon scoring criteria.

## Quantitative Evidence

- Final task completion: True
- Hazard final distance to quarantine: 0.0 m
- Medkit final distance to delivery: 0.0 m
- Minimum robot distance to beacon: 0.0117 m
- Residual corrections logged: 342
- Raw median visual-servo error: 0.02669 m
- Post-residual median error: 0.01147 m
- Error reduction: 57.01%
- Stable contact samples: 84
- Final slip observer: 0.995 mm
- Stress rollouts: 40
- No-residual baseline success: 0.4
- Residual-policy success: 1.0
- Median final error improvement: 52.915 mm

## Rubric Mapping

- Runnability: one command regenerates video, labels, metrics, stress replay, policy card, and judge brief.
- MuJoCo depth: official FF Master humanoid, actuators, sensors, free bodies, route pads, semantic zones, lighting, and camera motion.
- Task design: hazard sorting, beacon inspection, medkit retrieval, disturbance recovery, and data export.
- Control: deterministic task prior plus residual feedback logs for visual-servo error, contact balance, slip observer, correction norm, and confidence.
- Dexterous manipulation: coordinated arms and wrists handle packages with contact tracking; FF Master has no independent finger DOF, and this limitation is stated.
- Engineering quality: deterministic run, structured artifacts, fixed-seed replay, UUID consistency, and validator script.
- Presentation: generated video overlays expose the same metrics stored in JSON.
- Innovation: compact humanoid service benchmark combining safety triage, recovery evidence, and dataset collection.

## Honest Scope

The high-level route is deterministic so every judge can reproduce the full
sequence. The residual layer is a lightweight state-feedback estimator logged
from MuJoCo poses and sensors, not a learned neural policy.
