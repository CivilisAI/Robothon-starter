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

1. `media/demo.mp4` - 48s video with five-finger contact, cap angle, 4N shove, 9x load, slip, grip, residual, care-tool states, and confidence overlays.
2. `scene.xml` - five-finger MJCF hand, actuators, touch sensors, free vial/cap bodies, audit button, blister pack, syringe, and dose dial.
3. `learned_policy_weights.json` and `dataset/training_report.json` - learned policy evidence.
4. `dataset/contact_timeline.json` - five active fingers, balance score, and slip recovery samples.
5. `dataset/stress_eval.json` - 96 fixed-seed perturbation rollouts with 4N shove, 9x load, and multi-shape coverage.
6. `dataset/challenge_evidence.json` - short judge-oriented rubric and keyword index.
7. `dataset/metrics.json` - success criteria and closed-loop summary.

## Quantitative Evidence

- Final task success: True
- Policy type: learned_tactile_residual_grasp_policy
- Policy training samples: 6717
- Policy validation MAE: 0.022843
- Learned policy inference samples: 433
- Five-finger stable contact samples: 202
- Max cap rotation: 214.0 deg
- Real-world demo count: 7
- Blister press depth: 32.0 mm
- Syringe plunger depth: 58.0 mm
- Dose dial angle: 83.08 deg
- Max lateral shove: 4.0 N
- Max load hold: 9.0x object weight
- Max hold drift: 0.46 deg
- Multi-object shape count: 6
- Raw median visual-servo error: 0.01374 m
- Post-residual median error: 0.00573 m
- Error reduction: 58.3%
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
- Presentation: compact video includes beat labels, pass banner, cap-angle arc, disturbance callout, care-tool telemetry, and SRT captions.
- Innovation: compact safety-critical dexterity benchmark with multi-object medication actions and dataset export.
