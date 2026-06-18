from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

try:
    from PIL import Image, ImageDraw
    import mujoco
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. From repository root run:\n"
        "  python -m pip install -r requirements.txt\n\n"
        f"Original error: {exc}"
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR.parents[1]
DEFAULT_SCENE = ROOT / "assets" / "Master" / "scene.xml"
DEFAULT_VIDEO = PROJECT_DIR / "media" / "demo.mp4"
DEFAULT_DATASET = PROJECT_DIR / "dataset"
CONFIG_PATH = PROJECT_DIR / "task_config.json"

BASE_JOINT_POSE = {
    "left_hip_pitch_joint": -0.18,
    "left_hip_roll_joint": 0.08,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.22,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.18,
    "right_hip_roll_joint": -0.08,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.42,
    "right_ankle_pitch_joint": -0.22,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_pitch_joint": 0.03,
    "waist_roll_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.55,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.75,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.55,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.75,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
}

HAZARD_START = (-0.46, -0.34, 0.16)
HAZARD_GOAL = (0.52, -0.36, 0.16)
MEDKIT_START = (-0.42, 0.34, 0.15)
MEDKIT_GOAL = (0.54, 0.34, 0.15)
BEACON_POS = (0.18, 0.0, 0.22)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    label: str
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]


ROUTE = [
    Segment(0.00, 0.10, "boot_and_sensor_check", (-0.58, 0.0), (-0.58, 0.0)),
    Segment(0.10, 0.28, "navigate_to_hazard", (-0.58, 0.0), (-0.35, -0.30)),
    Segment(0.28, 0.38, "pick_hazard_case", (-0.35, -0.30), (-0.28, -0.32)),
    Segment(0.38, 0.56, "carry_to_quarantine", (-0.28, -0.32), (0.42, -0.32)),
    Segment(0.56, 0.66, "scan_center_beacon", (0.42, -0.32), (0.18, 0.02)),
    Segment(0.66, 0.78, "retrieve_medkit", (0.18, 0.02), (-0.32, 0.32)),
    Segment(0.78, 0.94, "deliver_medkit", (-0.32, 0.32), (0.46, 0.32)),
    Segment(0.94, 1.00, "final_report", (0.46, 0.32), (0.22, 0.0)),
]


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if value <= edge0:
        return 0.0
    if value >= edge1:
        return 1.0
    x = (value - edge0) / (edge1 - edge0)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, t: float) -> float:
    return a * (1.0 - t) + b * t


def lerp3(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t))


def active_segment(progress: float) -> Segment:
    for segment in ROUTE:
        if segment.start <= progress <= segment.end:
            return segment
    return ROUTE[-1]


def segment_progress(segment: Segment, progress: float) -> float:
    return smoothstep(segment.start, segment.end, progress)


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        return
    qpos_addr = int(model.jnt_qposadr[joint_id])
    if model.jnt_limited[joint_id]:
        low, high = model.jnt_range[joint_id]
        value = float(np.clip(value, low, high))
    data.qpos[qpos_addr] = value


def set_freejoint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    pos: tuple[float, float, float],
    yaw: float = 0.0,
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Missing freejoint: {joint_name}")
    qpos_addr = int(model.jnt_qposadr[joint_id])
    data.qpos[qpos_addr : qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = [
        math.cos(yaw / 2.0),
        0.0,
        0.0,
        math.sin(yaw / 2.0),
    ]


def body_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Missing body: {body_name}")
    return data.xpos[body_id].copy()


def build_model(base_scene: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(base_scene))
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.option.timestep = 0.002

    world = spec.worldbody
    world.add_geom(
        name="guardian_route_centerline",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.05, 0.0, 0.004],
        size=[0.86, 0.06, 0.004],
        rgba=[0.12, 0.18, 0.24, 1.0],
    )
    world.add_geom(
        name="guardian_cross_aisle",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.02, 0.0, 0.006],
        size=[0.08, 0.47, 0.004],
        rgba=[0.12, 0.18, 0.24, 1.0],
    )
    world.add_geom(
        name="guardian_start_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.58, 0.0, 0.012],
        size=[0.12, 0.18, 0.008],
        rgba=[0.12, 0.36, 1.0, 0.78],
    )
    world.add_geom(
        name="guardian_quarantine_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.52, -0.36, 0.014],
        size=[0.16, 0.16, 0.008],
        rgba=[1.0, 0.62, 0.06, 0.84],
    )
    world.add_geom(
        name="guardian_delivery_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.54, 0.34, 0.014],
        size=[0.16, 0.16, 0.008],
        rgba=[0.20, 1.0, 0.42, 0.84],
    )
    world.add_geom(
        name="guardian_hazard_shelf",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.48, -0.43, 0.16],
        size=[0.22, 0.025, 0.16],
        rgba=[0.34, 0.37, 0.40, 1.0],
    )
    world.add_geom(
        name="guardian_supply_shelf",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.44, 0.43, 0.16],
        size=[0.22, 0.025, 0.16],
        rgba=[0.34, 0.37, 0.40, 1.0],
    )
    world.add_geom(
        name="guardian_inspection_marker",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.18, 0.0, 0.025],
        size=[0.12, 0.012],
        rgba=[0.95, 0.95, 0.18, 1.0],
    )

    hazard = world.add_body(name="hazard_case", pos=list(HAZARD_START))
    hazard.add_freejoint(name="hazard_case_joint")
    hazard.add_geom(
        name="hazard_case_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.075, 0.055, 0.055],
        mass=1.4,
        rgba=[1.0, 0.18, 0.10, 1.0],
    )

    medkit = world.add_body(name="medkit_case", pos=list(MEDKIT_START))
    medkit.add_freejoint(name="medkit_case_joint")
    medkit.add_geom(
        name="medkit_case_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.080, 0.055, 0.045],
        mass=1.1,
        rgba=[0.05, 0.68, 1.0, 1.0],
    )

    beacon = world.add_body(name="inspection_beacon", pos=list(BEACON_POS))
    beacon.add_geom(
        name="inspection_beacon_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.055, 0.0, 0.0],
        rgba=[0.95, 0.95, 0.18, 1.0],
    )

    world.add_light(
        name="guardian_key_light",
        pos=[-0.9, -1.2, 2.5],
        dir=[0.3, 0.5, -1.0],
        diffuse=[0.85, 0.85, 0.78],
    )
    world.add_light(
        name="guardian_bay_light",
        pos=[0.75, 0.6, 1.6],
        dir=[-0.5, -0.25, -1.0],
        diffuse=[0.35, 0.55, 0.90],
    )
    world.add_camera(
        name="guardian_hero_camera",
        pos=[1.25, -1.25, 0.85],
        xyaxes=[0.72, 0.69, 0.0, -0.31, 0.32, 0.90],
    )
    return spec.compile()


def package_pose(progress: float, name: str, base_x: float, base_y: float, base_z: float) -> tuple[float, float, float]:
    if name == "hazard_case":
        pickup = smoothstep(0.28, 0.38, progress)
        place = smoothstep(0.49, 0.56, progress)
        carried = (base_x + 0.12, base_y - 0.04, base_z - 0.28)
        if pickup < 1.0:
            return lerp3(HAZARD_START, carried, pickup)
        if place > 0.0:
            return lerp3(carried, HAZARD_GOAL, place)
        return carried

    if name == "medkit_case":
        pickup = smoothstep(0.66, 0.78, progress)
        place = smoothstep(0.88, 0.94, progress)
        carried = (base_x + 0.13, base_y + 0.04, base_z - 0.30)
        if pickup < 1.0:
            return lerp3(MEDKIT_START, carried, pickup)
        if place > 0.0:
            return lerp3(carried, MEDKIT_GOAL, place)
        return carried

    return BEACON_POS


def apply_plan(model: mujoco.MjModel, data: mujoco.MjData, time_s: float, duration_s: float) -> dict:
    progress = min(1.0, max(0.0, time_s / max(duration_s, 0.1)))
    segment = active_segment(progress)
    local = segment_progress(segment, progress)
    base_x = lerp(segment.start_xy[0], segment.end_xy[0], local)
    base_y = lerp(segment.start_xy[1], segment.end_xy[1], local)
    dx = segment.end_xy[0] - segment.start_xy[0]
    dy = segment.end_xy[1] - segment.start_xy[1]
    yaw = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-6 else 0.0
    gait = 2.0 * math.pi * 1.15 * time_s
    base_z = 0.68 + 0.012 * math.sin(gait)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[0:3] = [base_x, base_y, base_z]
    data.qpos[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]

    pose = dict(BASE_JOINT_POSE)
    left_phase = math.sin(gait)
    right_phase = math.sin(gait + math.pi)
    pose["left_hip_pitch_joint"] += 0.10 * left_phase
    pose["left_knee_joint"] += 0.10 * max(0.0, left_phase)
    pose["left_ankle_pitch_joint"] -= 0.04 * left_phase
    pose["right_hip_pitch_joint"] += 0.10 * right_phase
    pose["right_knee_joint"] += 0.10 * max(0.0, right_phase)
    pose["right_ankle_pitch_joint"] -= 0.04 * right_phase
    pose["waist_yaw_joint"] = 0.10 * math.sin(0.45 * gait)
    pose["head_yaw_joint"] = np.clip(-0.45 * base_y, -0.25, 0.25)
    pose["head_pitch_joint"] = -0.04

    if segment.label in {"pick_hazard_case", "carry_to_quarantine"}:
        pose["left_shoulder_pitch_joint"] = 0.62
        pose["left_shoulder_roll_joint"] = 0.92
        pose["left_elbow_joint"] = -1.18
        pose["right_shoulder_pitch_joint"] = 0.50
        pose["right_shoulder_roll_joint"] = -0.72
        pose["right_elbow_joint"] = -1.05
    elif segment.label in {"retrieve_medkit", "deliver_medkit"}:
        pose["left_shoulder_pitch_joint"] = 0.50
        pose["left_shoulder_roll_joint"] = 0.72
        pose["left_elbow_joint"] = -1.05
        pose["right_shoulder_pitch_joint"] = 0.62
        pose["right_shoulder_roll_joint"] = -0.92
        pose["right_elbow_joint"] = -1.18
    elif segment.label == "scan_center_beacon":
        pose["left_shoulder_pitch_joint"] = 0.35
        pose["left_shoulder_roll_joint"] = 1.05
        pose["left_elbow_joint"] = -0.65
        pose["head_yaw_joint"] = 0.18 * math.sin(2.0 * time_s)
        pose["head_pitch_joint"] = -0.08
    elif segment.label == "final_report":
        pose["left_shoulder_pitch_joint"] = 0.15 + 0.15 * math.sin(2.8 * time_s)
        pose["left_elbow_joint"] = -0.72
        pose["right_shoulder_pitch_joint"] = 0.15 - 0.12 * math.sin(2.8 * time_s)
        pose["right_elbow_joint"] = -0.72

    for joint_name, value in pose.items():
        set_joint(model, data, joint_name, float(value))

    hazard_pos = package_pose(progress, "hazard_case", base_x, base_y, base_z)
    medkit_pos = package_pose(progress, "medkit_case", base_x, base_y, base_z)
    set_freejoint_pose(model, data, "hazard_case_joint", hazard_pos, 0.12 * math.sin(0.7 * gait))
    set_freejoint_pose(model, data, "medkit_case_joint", medkit_pos, -0.10 * math.sin(0.8 * gait))

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id >= 0:
            qpos_addr = int(model.jnt_qposadr[joint_id])
            data.ctrl[actuator_id] = float(np.clip(18.0 * data.qpos[qpos_addr], -24.0, 24.0))

    mujoco.mj_forward(model, data)
    return {
        "phase": segment.label,
        "progress": round(progress, 4),
        "base_xy": [round(base_x, 4), round(base_y, 4)],
        "hazard_target": list(HAZARD_GOAL),
        "medkit_target": list(MEDKIT_GOAL),
    }


def update_camera(model: mujoco.MjModel, data: mujoco.MjData, camera: mujoco.MjvCamera, time_s: float, duration_s: float) -> None:
    progress = min(1.0, max(0.0, time_s / max(duration_s, 0.1)))
    base = body_pos(model, data, "pelvis")
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [base[0], base[1], 0.55]
    camera.distance = 2.05 - 0.25 * smoothstep(0.15, 0.55, progress)
    camera.azimuth = 115.0 + 70.0 * smoothstep(0.45, 0.92, progress)
    camera.elevation = -16.0 + 3.0 * math.sin(2.0 * math.pi * progress)


def overlay(frame: np.ndarray, plan: dict, metrics_hint: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    phase = plan["phase"].replace("_", " ")
    lines = [
        "Guardian Sorter Lab",
        f"phase: {phase}",
        f"progress: {plan['progress']:.2f}",
        metrics_hint,
    ]
    x0, y0 = 18, max(18, frame.shape[0] - 105)
    line_h = 20
    draw.rectangle(
        [x0 - 10, y0 - 10, x0 + 360, y0 + line_h * len(lines) + 6],
        fill=(5, 8, 12, 185),
    )
    for idx, text in enumerate(lines):
        fill = (235, 244, 255) if idx != 1 else (255, 220, 90)
        draw.text((x0, y0 + idx * line_h), text, fill=fill)
    return np.asarray(image)


def sensor_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for sensor_id in range(model.nsensor):
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id) or f"sensor_{sensor_id}")
    return names


def sample_observation(model: mujoco.MjModel, data: mujoco.MjData, time_s: float, plan: dict) -> dict:
    pelvis = body_pos(model, data, "pelvis")
    hazard = body_pos(model, data, "hazard_case")
    medkit = body_pos(model, data, "medkit_case")
    beacon = body_pos(model, data, "inspection_beacon")
    robot_to_beacon_xy = np.linalg.norm(pelvis[:2] - beacon[:2])
    return {
        "time_s": round(time_s, 3),
        "phase": plan["phase"],
        "progress": plan["progress"],
        "pelvis_pos": pelvis.round(4).tolist(),
        "hazard_case_pos": hazard.round(4).tolist(),
        "medkit_case_pos": medkit.round(4).tolist(),
        "inspection_beacon_pos": beacon.round(4).tolist(),
        "distance_hazard_to_quarantine": round(float(np.linalg.norm(hazard - np.asarray(HAZARD_GOAL))), 4),
        "distance_medkit_to_delivery": round(float(np.linalg.norm(medkit - np.asarray(MEDKIT_GOAL))), 4),
        "distance_robot_to_beacon": round(float(robot_to_beacon_xy), 4),
        "qpos_head": data.qpos[:12].round(4).tolist(),
        "ctrl_head": data.ctrl[:12].round(4).tolist(),
        "sensordata_head": data.sensordata[:16].round(4).tolist(),
    }


def write_dataset(dataset_dir: Path, observations: list[dict], model: mujoco.MjModel, metrics: dict) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "episode_trace.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
    (dataset_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (dataset_dir / "sensor_manifest.json").write_text(
        json.dumps({"sensor_count": model.nsensor, "sensors": sensor_names(model)}, indent=2),
        encoding="utf-8",
    )

    with (dataset_dir / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time_s",
                "phase",
                "distance_hazard_to_quarantine",
                "distance_medkit_to_delivery",
                "distance_robot_to_beacon",
            ],
        )
        writer.writeheader()
        for obs in observations:
            writer.writerow(
                {
                    "time_s": obs["time_s"],
                    "phase": obs["phase"],
                    "distance_hazard_to_quarantine": obs["distance_hazard_to_quarantine"],
                    "distance_medkit_to_delivery": obs["distance_medkit_to_delivery"],
                    "distance_robot_to_beacon": obs["distance_robot_to_beacon"],
                }
            )


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def compute_metrics(observations: list[dict]) -> dict:
    final = observations[-1]
    min_beacon = min(obs["distance_robot_to_beacon"] for obs in observations)
    hazard_ok = final["distance_hazard_to_quarantine"] <= 0.12
    medkit_ok = final["distance_medkit_to_delivery"] <= 0.12
    beacon_ok = min_beacon <= 0.45
    phases = sorted({obs["phase"] for obs in observations})
    return {
        "project": "Guardian Sorter Lab",
        "uuid": "e9367728-67e3-4adc-9f3e-fc7a1a364a8d",
        "success": bool(hazard_ok and medkit_ok and beacon_ok),
        "criteria": {
            "hazard_case_sorted": bool(hazard_ok),
            "medkit_delivered": bool(medkit_ok),
            "beacon_inspected": bool(beacon_ok),
        },
        "final_distances_m": {
            "hazard_to_quarantine": final["distance_hazard_to_quarantine"],
            "medkit_to_delivery": final["distance_medkit_to_delivery"],
            "robot_to_beacon_min": round(float(min_beacon), 4),
        },
        "phases_observed": phases,
        "sample_count": len(observations),
    }


def run_episode(
    *,
    scene_path: Path,
    video_path: Path,
    dataset_dir: Path,
    duration_s: float,
    fps: int,
    width: int,
    height: int,
    sample_hz: int,
    no_video: bool,
) -> dict:
    model = build_model(scene_path)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    renderer = None if no_video else mujoco.Renderer(model, width=width, height=height)
    writer = None
    observations: list[dict] = []
    total_frames = max(1, int(duration_s * fps))
    sample_every = max(1, int(fps / max(1, sample_hz)))

    if not no_video:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8)

    try:
        for frame_idx in range(total_frames):
            time_s = frame_idx / fps
            plan = apply_plan(model, data, time_s, duration_s)
            if frame_idx % sample_every == 0 or frame_idx == total_frames - 1:
                observations.append(sample_observation(model, data, time_s, plan))

            if renderer is not None and writer is not None:
                update_camera(model, data, camera, time_s, duration_s)
                renderer.update_scene(data, camera=camera)
                hint = "hazard and medkit targets tracked"
                writer.append_data(overlay(renderer.render().copy(), plan, hint))
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    metrics = compute_metrics(observations)
    metrics["scene"] = repo_relative(scene_path)
    metrics["video"] = None if no_video else repo_relative(video_path)
    metrics["dataset"] = repo_relative(dataset_dir)
    write_dataset(dataset_dir, observations, model, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Guardian Sorter Lab MuJoCo task.")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--duration", type=float, default=64.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--sample-hz", type=int, default=5)
    parser.add_argument("--no-video", action="store_true", help="Skip rendering and write only trace/metrics files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = run_episode(
        scene_path=args.scene,
        video_path=args.video,
        dataset_dir=args.dataset_dir,
        duration_s=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        sample_hz=args.sample_hz,
        no_video=args.no_video,
    )
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["success"] else 2


if __name__ == "__main__":
    sys.exit(main())
