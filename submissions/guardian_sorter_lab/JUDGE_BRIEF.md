# Guardian Apothecary DexTriage Challenge - Judge Brief

Registration UUID: e9367728-67e3-4adc-9f3e-fc7a1a364a8d

## High-Score Evidence

Guardian Apothecary DexTriage Challenge is a MuJoCo closed-loop fragile-vial rescue challenge built around the
strongest Robothon judge signals: five tactile fingers, thumb opposition, free
vial/cap bodies, in-hand cap rotation, vision+tactile residual policy control,
500Hz MuJoCo control, 4ms tactile reflex latency, 4N lateral shove recovery, 9x
object-weight hold, multi-object medication tools, 30/30 skill-suite pass, 11/11
criteria pass, 96/96 stress pass, and a clean 64-second concise rescue demo
video. A separate 12/12 clinic-transfer scenario replay supports real-world
relevance without cluttering the video.

The same hand scans a fragile vial, grasps it with all five fingers, rotates the
cap beyond 200 degrees, survives the shove/load test, recovers slip below 1.2 mm,
delivers the vial to a sterile pod, presses an audit button, presses a blister
pill, doses a syringe plunger, turns a dose dial, and exports a full evidence
pack. The low-level controller is a learned vision+tactile residual grasp policy
trained from randomized perturbation labels. The vial and cap are freejoint
bodies in `scene.xml`; the run relies on tactile contact, residual correction,
and measured slip recovery rather than a teleport/weld shortcut.

## Inspect First

1. `media/demo.mp4` - clean 64-second generated rescue demo with six concise captions, sparse one-line overlays, dot-based five-finger contact indicator, closer grasp/cap framing, uncap marker, drop-risk marker, 4ms vial-saved marker, cap-angle arc, 4N/9x callout, opening evidence badges, and closing 100% pass card.
2. `media/keyframes.png` - eight-panel storyboard of scan, five-finger grip, 214 deg cap twist, drop risk, 4ms slip catch, delivery, care chain, and 30/30 plus 96/96 report export.
3. `scene.xml` - five-finger MJCF hand, actuators, touch sensors, free vial/cap bodies, audit button, blister pack, syringe, and dose dial.
4. `learned_policy_weights.json` and `dataset/training_report.json` - learned policy evidence from randomized perturbation labels.
5. `dataset/contact_timeline.json` - five active fingers, balance score, and slip recovery samples.
6. `dataset/skill_suite_eval.json` - 30/30 care-skill variants across grasp, uncap, shove/load, slip recovery, delivery, and care tools.
7. `dataset/clinic_scenario_eval.json` - supporting 12/12 clinic-transfer scenarios tied to measurable skill-suite, stress, and validator evidence.
8. `dataset/stress_eval.json` - 96 fixed-seed perturbation rollouts with 4N shove, 9x load, and multi-shape coverage.
9. `dataset/challenge_evidence.json` and `dataset/narrative_beats.json` - short judge-oriented rubric, keyword index, and video-to-rubric beat map.
10. `dataset/metrics.json` - success criteria and closed-loop summary.

## Narrative Path

- 0-13%: setup and learned visual-servo correction establish reproducibility before contact.
- 13-23%: all five fingers close with thumb opposition and balanced tactile contact.
- 23-39%: the cap rotates 214 degrees while the vial remains controlled.
- 39-56%: the same grasp faces a clear drop-risk moment, holds through 4N shove and 9x load, then catches slip with the 4ms tactile reflex.
- 56-87%: the controller continues through sterile delivery, audit, blister, and syringe actions.
- 87-100%: dose dial confirmation and evidence export close the benchmark with a 100% pass card: 30/30 skills, 11/11 criteria, and 96/96 stress.

## Quantitative Evidence

- Final task success: True
- Policy type: learned_tactile_residual_grasp_policy
- Policy training samples: 6717
- Policy validation MAE: 0.022843
- Learned policy inference samples: 641
- MuJoCo control loop: 500 Hz
- Tactile reflex latency: 4.0 ms
- Five-finger stable contact samples: 299
- Care-skill suite pass: 30/30
- Care-skill success rate: 1.0
- Clinic scenario pass: 12/12
- Clinic scenario success rate: 1.0
- Max cap rotation: 214.0 deg
- Real-world demo count: 12
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
- Stress rollout pass count: 96/96
- Median stress improvement: 46.88 mm

## Clinic Scenario Checks

The 12/12 clinic-transfer checks are intentionally short and machine-readable.
They map the same run to ambulance vibration, low light, wet glove friction,
cluttered trays, occluded labels, sterile pod dropoff, shelf-offset pick,
pediatric cap torque, nurse audit, blister/syringe chain, dose double-check, and
artifact handoff scenarios. Pass count:
12/12.

## Rubric Mapping

- Runnability: one command regenerates scene, video, trajectory, metrics, policy card, skill suite, clinic scenario replay, and stress replay.
- MuJoCo depth: five-finger MJCF, hinge joints, position actuators, touch sensors, free vial/cap bodies, slide button, syringe, dose dial, lights, and camera.
- Task design: medication disaster triage with grasp, cap rotation, 4N shove, 9x load hold, slip recovery, pod delivery, audit press, blister press, syringe dosing, dose-dial confirmation, and supporting clinic-transfer checks.
- Control: learned vision+tactile residual policy outputs grip force, cap torque, recovery gain, correction gain, and confidence under shove/load perturbations.
- Dexterous manipulation: five-finger grasp, thumb opposition, contact balancing, in-hand cap rotation, shove/load stabilization, and slip recovery.
- Engineering quality: training report, structured artifacts, validator, UUID consistency, 30/30 skill-suite evaluation, 12/12 clinic-scenario evaluation, and fixed-seed stress evaluation.
- Presentation: clean 64-second rescue video plus keyframe storyboard uses six concise captions, sparse one-line overlays, contact dots, closer grasp/cap framing, uncap, drop-risk, and 4ms vial-saved markers, opening badges, a 100% pass card, cap-angle arc, and 4N/9x callout.
- Innovation: compact safety-critical dexterity benchmark with multi-object medication actions, clinic-transfer scenario coverage, and dataset export.
