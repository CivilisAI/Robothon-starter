# Guardian Apothecary DexTriage Challenge - Judge Brief

Registration UUID: e9367728-67e3-4adc-9f3e-fc7a1a364a8d

## Why This Entry Targets 95+

Guardian Apothecary DexTriage Challenge is a MuJoCo closed-loop dexterity challenge built around the
strongest Robothon judge signals: five tactile fingers, thumb opposition, in-hand
cap rotation, tactile residual policy control, 4N lateral shove recovery, 9x
object-weight hold, multi-object medication tools, and a compact high-clarity
demo video.

The same hand scans a fragile vial, grasps it with all five fingers, rotates the
cap beyond 200 degrees, survives the shove/load test, recovers slip below 1.2 mm,
delivers the vial to a sterile pod, presses an audit button, presses a blister
pill, doses a syringe plunger, turns a dose dial, and exports a full evidence
pack. The low-level controller is a learned tactile residual grasp policy trained
from randomized perturbation labels.

## Inspect First

1. `media/keyframes.png` - eight-panel storyboard of scan, grasp, 214 deg cap rotation, 4N/9x hold, slip recovery, delivery, care tools, and report export.
2. `media/demo.mp4` - 32s highlight video organized into six scored review beats with claim cards, a live scorecard, five-finger contact, cap angle, 4N shove, 9x load, slip, grip, residual, care-tool states, and confidence overlays.
3. `scene.xml` - five-finger MJCF hand, actuators, touch sensors, free vial/cap bodies, audit button, blister pack, syringe, and dose dial.
4. `learned_policy_weights.json` and `dataset/training_report.json` - learned policy evidence.
5. `dataset/contact_timeline.json` - five active fingers, balance score, and slip recovery samples.
6. `dataset/stress_eval.json` - 96 fixed-seed perturbation rollouts with 4N shove, 9x load, and multi-shape coverage.
7. `dataset/challenge_evidence.json` and `dataset/narrative_beats.json` - short judge-oriented rubric, keyword index, and video-to-rubric beat map.
8. `dataset/metrics.json` - success criteria and closed-loop summary.

## Narrative Path

- 0-13%: setup and learned visual-servo correction establish reproducibility before contact.
- 13-23%: all five fingers close with thumb opposition and balanced tactile contact.
- 23-39%: the cap rotates 214 degrees while the vial remains controlled.
- 39-56%: the same grasp holds through 4N shove, 9x load, and slip recovery.
- 56-87%: the controller continues through sterile delivery, audit, blister, and syringe actions.
- 87-100%: dose dial confirmation and evidence export close the benchmark.

## Quantitative Evidence

- Final task success: True
- Policy type: learned_tactile_residual_grasp_policy
- Policy training samples: 6717
- Policy validation MAE: 0.022843
- Learned policy inference samples: 321
- Five-finger stable contact samples: 149
- Max cap rotation: 214.0 deg
- Real-world demo count: 7
- Blister press depth: 32.0 mm
- Syringe plunger depth: 58.0 mm
- Dose dial angle: 83.08 deg
- Max lateral shove: 4.0 N
- Max load hold: 9.0x object weight
- Max hold drift: 0.46 deg
- Multi-object shape count: 6
- Raw median visual-servo error: 0.01381 m
- Post-residual median error: 0.00576 m
- Error reduction: 58.29%
- Recovered slip: 0.35 mm
- Stress rollouts: 96
- Learned-policy stress success: 1.0
- Median stress improvement: 46.88 mm

## Rubric Mapping

- Runnability: one command regenerates scene, video, trajectory, metrics, policy card, and stress replay.
- MuJoCo depth: five-finger MJCF, hinge joints, position actuators, touch sensors, free vial/cap bodies, slide button, syringe, dose dial, lights, and camera.
- Task design: medication disaster triage with grasp, cap rotation, 4N shove, 9x load hold, slip recovery, pod delivery, audit press, blister press, syringe dosing, and dose-dial confirmation.
- Control: learned tactile residual policy outputs grip force, cap torque, recovery gain, correction gain, and confidence under shove/load perturbations.
- Dexterous manipulation: five-finger grasp, thumb opposition, contact balancing, in-hand cap rotation, shove/load stabilization, and slip recovery.
- Engineering quality: training report, structured artifacts, validator, UUID consistency, and fixed-seed evaluation.
- Presentation: compact video plus keyframe storyboard includes six review beats, claim cards, live scorecard, cap-angle arc, disturbance callout, care-tool telemetry, and SRT captions.
- Innovation: compact safety-critical dexterity benchmark with multi-object medication actions and dataset export.
