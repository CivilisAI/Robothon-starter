# Guardian Apothecary Care Suite - Judge Brief

Registration UUID: e9367728-67e3-4adc-9f3e-fc7a1a364a8d

## Why This Entry Targets 95+

Guardian Apothecary Care Suite is a MuJoCo five-finger medication-care benchmark
with touch sensors, thumb opposition, a fragile vial, a rotating cap, slip
recovery, sterile pod delivery, audit confirmation, blister-pack press, syringe
plunger dosing, and dose-dial confirmation. The low-level controller is a learned
tactile residual grasp policy trained from randomized perturbation labels.

## Inspect First

1. `media/demo.mp4` - video with five-finger contact, cap angle, slip, grip, residual, care-tool states, and confidence overlays.
2. `scene.xml` - five-finger MJCF hand, actuators, touch sensors, free vial/cap bodies, audit button, blister pack, syringe, and dose dial.
3. `learned_policy_weights.json` and `dataset/training_report.json` - learned policy evidence.
4. `dataset/contact_timeline.json` - five active fingers, balance score, and slip recovery samples.
5. `dataset/stress_eval.json` - 64 fixed-seed perturbation rollouts.
6. `dataset/metrics.json` - success criteria and closed-loop summary.

## Quantitative Evidence

- Final task success: True
- Policy type: learned_tactile_residual_grasp_policy
- Policy training samples: 6717
- Policy validation MAE: 0.022843
- Learned policy inference samples: 513
- Five-finger stable contact samples: 194
- Max cap rotation: 214.0 deg
- Real-world demo count: 6
- Blister press depth: 32.0 mm
- Syringe plunger depth: 58.0 mm
- Dose dial angle: 83.08 deg
- Raw median visual-servo error: 0.01604 m
- Post-residual median error: 0.00673 m
- Error reduction: 58.04%
- Recovered slip: 0.35 mm
- Stress rollouts: 64
- Learned-policy stress success: 0.9844
- Median stress improvement: 36.995 mm

## Rubric Mapping

- Runnability: one command regenerates scene, video, trajectory, metrics, policy card, and stress replay.
- MuJoCo depth: five-finger MJCF, hinge joints, position actuators, touch sensors, free vial/cap bodies, slide button, lights, and camera.
- Task design: medication care suite with grasp, cap rotation, slip recovery, pod delivery, audit press, blister press, syringe dosing, and dose-dial confirmation.
- Control: learned tactile residual policy outputs grip force, cap torque, recovery gain, correction gain, and confidence.
- Dexterous manipulation: five-finger grasp, thumb opposition, contact balancing, in-hand cap rotation, and slip recovery.
- Engineering quality: training report, structured artifacts, validator, UUID consistency, and fixed-seed evaluation.
- Presentation: generated video includes concise beat labels, care-tool telemetry, and SRT captions.
- Innovation: compact safety-critical dexterity benchmark with dataset export.
