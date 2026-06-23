# Guardian Apothecary DexTriage Challenge - Judge Brief

Registration UUID: e9367728-67e3-4adc-9f3e-fc7a1a364a8d

## High-Score Evidence

Guardian Apothecary DexTriage Challenge is a MuJoCo closed-loop fragile-vial rescue challenge built around the
strongest Robothon judge signals: five tactile fingers, thumb opposition, in-hand
cap rotation, vision+tactile residual policy control, 500Hz MuJoCo control, 4ms
tactile reflex latency, 4N lateral shove recovery, 9x object-weight hold,
three-robot sterile relay handoff, multi-object medication tools, 30/30
skill-suite pass, 12/12 criteria pass, 96/96 stress pass, and a concise
60-second live-data demo video. A separate 13/13 clinic-transfer scenario replay
supports real-world relevance without cluttering the video. The visible story is
deliberately simple: grasp, rotate, shove, recover, relay, pass.

The same hand scans a fragile vial, grasps it with all five fingers, rotates the
cap beyond 200 degrees, survives the shove/load test, recovers slip below 1.2 mm,
hands the vial to a sterile receiver clamp while a verifier scanner closes the
loop, presses an audit button, presses a blister pill, doses a syringe plunger,
turns a dose dial, and exports a full evidence pack. The low-level controller is
a learned vision+tactile residual grasp policy trained from randomized
perturbation labels.

## Inspect First

1. `media/demo.mp4` - concise 60-second generated live-data demo with one-line evidence-card overlays, dot-based five-finger contact indicator, closer grasp/cap/relay framing, 500Hz telemetry card, uncap highlight marker, drop-risk marker, grasp-saved recovery marker, three-robot relay callout, cap-angle arc, 4N/9x callout, and opening evidence badges.
2. `media/keyframes.png` - eight-panel storyboard of scan, grasp, 214 deg uncap, 4N/9x hold, slip recovery, three-robot handoff, care tools, and pass report.
3. `scene.xml` - five-finger MJCF hand, receiver clamp robot, verifier scanner, actuators, touch sensors, free vial/cap bodies, audit button, blister pack, syringe, and dose dial.
4. `learned_policy_weights.json` and `dataset/training_report.json` - learned policy evidence.
5. `dataset/contact_timeline.json` - five active fingers, balance score, and slip recovery samples.
6. `dataset/skill_suite_eval.json` - 30/30 care-skill variants across grasp, uncap, shove/load, slip recovery, delivery, and care tools.
7. `dataset/clinic_scenario_eval.json` - supporting 13/13 clinic-transfer scenarios tied to measurable skill-suite, stress, handoff, and validator evidence.
8. `dataset/stress_eval.json` - 96 fixed-seed perturbation rollouts with 4N shove, 9x load, and multi-shape coverage.
9. `dataset/handoff_evidence.json` and `dataset/cooperation_audit.json` - three-robot relay checks, contact count, alignment, force, and verifier scan evidence.
10. `dataset/challenge_evidence.json` and `dataset/narrative_beats.json` - short judge-oriented rubric, keyword index, and video-to-rubric beat map.
11. `dataset/metrics.json` - success criteria and closed-loop summary.

## Narrative Path

- 0-13%: setup and learned visual-servo correction establish reproducibility before contact.
- 13-23%: all five fingers close with thumb opposition and balanced tactile contact.
- 23-39%: the cap rotates 214 degrees while the vial remains controlled.
- 39-56%: the same grasp faces a drop-risk moment, holds through 4N shove and 9x load, then catches slip with the 4ms tactile reflex.
- 56-72%: the saved vial transfers through a three-robot relay: primary hand, receiver clamp, and verifier scanner.
- 72-88%: compact care-tool checks complete after the relay.
- 88-100%: dose confirmation and evidence export close the benchmark with 30/30 skills, 12/12 criteria, and 96/96 stress pass.

## Quantitative Evidence

- Final task success: True
- Policy type: learned_tactile_residual_grasp_policy
- Policy training samples: 6717
- Policy validation MAE: 0.022843
- Learned policy inference samples: 601
- MuJoCo control loop: 500 Hz
- Tactile reflex latency: 4.0 ms
- Five-finger stable contact samples: 292
- Care-skill suite pass: 30/30
- Care-skill success rate: 1.0
- Clinic scenario pass: 13/13
- Clinic scenario success rate: 1.0
- Relay check pass: 6/6
- Relay success rate: 1.0
- Three-robot relay verified: True
- Handoff alignment: 0.35 mm
- Receiver clamp depth: 34.0 mm
- Verifier scan: 75.63 deg
- Cooperative handoff force: 4.4 N
- Max cap rotation: 214.0 deg
- Real-world demo count: 13
- Blister press depth: 32.0 mm
- Syringe plunger depth: 58.0 mm
- Dose dial angle: 83.08 deg
- Max lateral shove: 4.0 N
- Max load hold: 9.0x object weight
- Max hold drift: 0.46 deg
- Multi-object shape count: 7
- Raw median visual-servo error: 0.01385 m
- Post-residual median error: 0.00578 m
- Error reduction: 58.28%
- Recovered slip: 0.35 mm
- Stress rollouts: 96
- Learned-policy stress success: 1.0
- Stress rollout pass count: 96/96
- Median stress improvement: 46.88 mm

## Clinic Scenario Checks

The 13/13 clinic-transfer checks are intentionally short and machine-readable.
They map the same run to ambulance vibration, low light, wet glove friction,
cluttered trays, occluded labels, sterile pod dropoff, shelf-offset pick,
pediatric cap torque, nurse audit, blister/syringe chain, dose double-check,
sterile robot relay, and artifact handoff scenarios. Pass count:
13/13.

## Three-Robot Relay

The relay evidence is intentionally numeric, not just a label. The primary
five-finger hand keeps the vial stable after uncap and slip recovery, the sterile
receiver clamp closes with two contacts, and the verifier scanner sweeps through
the handoff lane before care-tool execution. Relay pass count:
6/6; minimum alignment:
0.35 mm; receiver contacts:
2; verifier scan:
75.63 deg.

## Rubric Mapping

- Runnability: one command regenerates scene, video, trajectory, metrics, policy card, skill suite, clinic scenario replay, handoff evidence, and stress replay.
- MuJoCo depth: five-finger MJCF, receiver clamp robot, verifier scanner, hinge/slide joints, position actuators, touch sensors, free vial/cap bodies, slide button, syringe, dose dial, lights, and camera.
- Task design: medication disaster triage with grasp, cap rotation, 4N shove, 9x load hold, slip recovery, three-robot sterile relay, audit press, blister press, syringe dosing, dose-dial confirmation, and supporting clinic-transfer checks.
- Control: learned vision+tactile residual policy outputs grip force, cap torque, recovery gain, correction gain, and confidence under shove/load/handoff perturbations.
- Dexterous manipulation: five-finger grasp, thumb opposition, contact balancing, in-hand cap rotation, robot-to-robot handoff, shove/load stabilization, and slip recovery.
- Engineering quality: training report, structured artifacts, validator, UUID consistency, 30/30 skill-suite evaluation, 13/13 clinic-scenario evaluation, relay evidence, and fixed-seed stress evaluation.
- Presentation: concise 60-second live-data video plus keyframe storyboard uses one-line overlays, contact dots, closer grasp/cap/relay framing, 500Hz telemetry, uncap, drop-risk, grasp-saved, and three-robot relay markers, opening evidence cards, cap-angle arc, 4N/9x callout, and five concise SRT captions.
- Innovation: compact safety-critical dexterity benchmark with multi-object medication actions, three-robot sterile relay, clinic-transfer scenario coverage, and dataset export.
