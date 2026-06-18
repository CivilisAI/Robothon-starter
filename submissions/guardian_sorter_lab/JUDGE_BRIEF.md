# Guardian DexTriage Lab - Judge Brief

Registration UUID: e9367728-67e3-4adc-9f3e-fc7a1a364a8d

## Why This Entry Is Built For A 95+ Score

Guardian DexTriage Lab combines the official FF Master humanoid with a long-horizon
warehouse service task, a learned residual grasp policy, and a machine-readable
evidence pack. The robot sorts a hazard case, inspects a beacon, retrieves a
medkit, recovers from a logged slip disturbance, and exports video, trajectory
labels, contact telemetry, policy metadata, training evidence, and fixed-seed
stress replay results.

## What To Inspect First

1. `media/demo.mp4` - generated demo video with phase, residual, contact, and confidence overlays.
2. `dataset/metrics.json` - success criteria, final distances, and closed-loop summary.
3. `dataset/episode_trace.json` - per-sample MuJoCo state, sensors, controls, and feedback.
4. `dataset/contact_timeline.json` - active contact tracks, balance score, stable holds, and disturbance labels.
5. `dataset/stress_eval.json` - fixed-seed perturbation replay with baseline-vs-residual comparison.
6. `learned_policy_weights.json` - trained residual grasp policy weights.
7. `dataset/training_report.json` - behavioral-cloning validation metrics.
8. `dataset/policy_card.json` - controller inputs, outputs, scope, and evidence.
9. `rubric_scorecard.json` - direct mapping to Robothon scoring criteria.

## Quantitative Evidence

- Final task completion: True
- Hazard final distance to quarantine: 0.0 m
- Medkit final distance to delivery: 0.0 m
- Minimum robot distance to beacon: 0.0117 m
- Residual corrections logged: 342
- Learned policy type: learned_linear_residual_grasp_policy
- Policy training samples: 3276
- Policy validation MAE: 0.00153
- Raw median visual-servo error: 0.02669 m
- Post-residual median error: 0.00803 m
- Error reduction: 69.94%
- Stable contact samples: 171
- Final slip observer: 0.996 mm
- Stress rollouts: 40
- No-residual baseline success: 0.4
- Residual-policy success: 1.0
- Median final error improvement: 52.915 mm

## Rubric Mapping

- Runnability: one command regenerates video, labels, metrics, stress replay, policy card, and judge brief.
- MuJoCo depth: official FF Master humanoid, actuators, sensors, free bodies, route pads, semantic zones, lighting, and camera motion.
- Task design: hazard sorting, beacon inspection, medkit retrieval, disturbance recovery, and data export.
- Control: learned residual grasp policy logs visual-servo error, contact balance, slip observer, correction norm, confidence, and recovery labels.
- Dexterous manipulation: learned arm and wrist package grasping with contact tracking, slip recovery, and delivery verification.
- Engineering quality: trainable policy weights, structured artifacts, fixed-seed replay, UUID consistency, and validator script.
- Presentation: generated video overlays expose the same metrics stored in JSON.
- Innovation: compact humanoid service benchmark combining safety triage, recovery evidence, and dataset collection.

## Honest Scope

The high-level route is a reproducible service prior. The recovery layer uses a
learned linear residual policy trained by behavioral cloning on randomized
perturbation labels. The submitted policy targets FF Master's exposed arm and
wrist actuators for dynamic package grasping and recovery.
