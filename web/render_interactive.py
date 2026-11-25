"""Render a trained 3DGS model on the selected GPU over a JSON lines pipe."""

import argparse
import base64
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np
from PIL import Image

from render_preview import model_degree


PREVIEW_SIZE = 672
HIGH_QUALITY_SIZE = 1440
HIGH_QUALITY_SUPERSAMPLING = 1.5
POSE_LIMITS = (
    ("yaw", -math.pi, math.pi),
    ("pitch", -1.25, 1.25),
    ("zoom", 0.35, 4.0),
    ("panX", -1.5, 1.5),
    ("panY", -1.5, 1.5),
)


@dataclass(frozen=True)
class CalibratedCamera:
    name: str
    position: np.ndarray
    rotation: np.ndarray
    width: int
    height: int
    fx: float
    fy: float


def load_cameras(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("相机文件格式无效。")
    cameras = []
    for item in data:
        try:
            camera = CalibratedCamera(
                str(item["img_name"]), np.asarray(item["position"], dtype=np.float64),
                np.asarray(item["rotation"], dtype=np.float64), int(item["width"]),
                int(item["height"]), float(item["fx"]), float(item["fy"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if (
            camera.position.shape == (3,) and camera.rotation.shape == (3, 3)
            and np.isfinite(camera.position).all() and np.isfinite(camera.rotation).all()
            and np.isfinite([camera.fx, camera.fy]).all()
            and min(camera.width, camera.height, camera.fx, camera.fy) > 0
            and np.linalg.norm(camera.rotation.T @ camera.rotation - np.eye(3)) < 0.1
        ):
            cameras.append(camera)
    if not cameras:
        raise ValueError("相机文件中没有可用的标定视角。")
    return sorted(cameras, key=lambda camera: camera.name)


def unit_vector(vector):
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("无法确定模型的相机方向。")
    return vector / norm


def focus_from_rays(cameras):
    if len(cameras) < 2:
        return None
    positions = np.stack([camera.position for camera in cameras])
    forwards = np.stack([unit_vector(camera.rotation[:, 2]) for camera in cameras])
    origin = np.median(positions, axis=0)
    positions = positions - origin
    projectors = np.eye(3) - forwards[:, :, None] * forwards[:, None, :]
    offsets = np.einsum("nij,nj->ni", projectors, positions)
    weights = np.ones(len(cameras))
    for _ in range(5):
        system = np.einsum("n,nij->ij", weights, projectors)
        target = weights @ offsets
        focus, _, rank, singular = np.linalg.lstsq(system, target, rcond=1e-5)
        if rank < 3 or singular[-1] < 1e-5 * singular[0]:
            return None
        errors = np.linalg.norm(np.einsum("nij,nj->ni", projectors, focus - positions), axis=1)
        scale = max(float(np.median(errors)) * 1.5, 1e-8)
        weights = np.minimum(1.0, scale / np.maximum(errors, 1e-8))
    depths = np.einsum("ni,ni->n", forwards, focus - positions)
    if np.median(depths) <= 0 or not np.isfinite(focus).all():
        return None
    return focus + origin


def orbit_focus(cameras, representative, gaussians):
    forward = unit_vector(representative.rotation[:, 2])
    focus = focus_from_rays(cameras)
    if focus is not None:
        offset = focus - representative.position
        distance = np.linalg.norm(offset)
        if distance > 0.2 and np.dot(offset, forward) > 0.7 * distance:
            return focus

    # Forward-facing cameras can have nearly parallel optical axes. Keep the
    # representative view and use visible Gaussian depths for the orbit radius.
    points = gaussians.get_xyz.detach()
    stride = max(math.ceil(len(points) / 8192), 1)
    offsets = points[::stride].cpu().numpy() - representative.position
    depths = offsets @ forward
    visible = depths[np.isfinite(depths) & (depths > 0.2)]
    if visible.size:
        distance = float(np.median(visible))
    else:
        positions = np.stack([camera.position for camera in cameras])
        distance = max(1.0, float(np.linalg.norm(np.median(positions, axis=0) - representative.position)))
    return representative.position + forward * distance


def read_pose(line):
    try:
        source = json.loads(line)
    except ValueError as exc:
        raise ValueError("相机姿态的 JSON 格式无效。") from exc
    if not isinstance(source, dict):
        raise ValueError("相机姿态必须是 JSON 对象。")
    pose = []
    for key, lower, upper in POSE_LIMITS:
        value = source.get(key)
        if type(value) not in (int, float) or not lower <= value <= upper:
            raise ValueError(f"相机参数 {key} 超出允许范围。")
        pose.append(float(value))
    quality = source.get("quality", "preview")
    if quality not in ("preview", "high"):
        raise ValueError("渲染画质设置无效。")
    pose.append(quality)
    return pose


class InteractiveRenderer:
    def __init__(self, model, cameras_path):
        import torch
        from gaussian_renderer import render
        from scene.cameras import MiniCam
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import focal2fov, getProjectionMatrix

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU 当前不可用。")
        self.torch = torch
        self.render = render
        self.make_camera = MiniCam
        cameras = load_cameras(cameras_path)
        front_cameras = [camera for camera in cameras if camera.name.startswith("front-forward/")]
        representative = front_cameras[len(front_cameras) * 4 // 5] if front_cameras else cameras[len(cameras) // 5]

        with torch.no_grad():
            gaussians = GaussianModel(model_degree(model))
            gaussians.load_ply(str(model))
            self.focus = orbit_focus(cameras, representative, gaussians)
            # The trained parameters are fixed. Keep the normalized rotations,
            # scales, opacity and joined SH tensor across camera poses.
            self.gaussians = SimpleNamespace(
                get_xyz=gaussians.get_xyz,
                get_rotation=gaussians.get_rotation,
                get_scaling=gaussians.get_scaling,
                get_opacity=gaussians.get_opacity,
                get_features=gaussians.get_features,
                active_sh_degree=gaussians.active_sh_degree,
            )

        offset = representative.position - self.focus
        self.distance = float(np.linalg.norm(offset))
        self.outward = unit_vector(offset)
        self.right = unit_vector(np.cross(-representative.rotation[:, 1], self.outward))
        self.up = np.cross(self.outward, self.right)
        self.dimensions = {}
        for quality, size in (("preview", PREVIEW_SIZE), ("high", HIGH_QUALITY_SIZE)):
            scale = size / max(representative.width, representative.height)
            self.dimensions[quality] = tuple(max(1, round(axis * scale))
                                             for axis in (representative.width, representative.height))
        self.fov_x = focal2fov(representative.fx, representative.width)
        self.fov_y = focal2fov(representative.fy, representative.height)
        self.near = max(0.0001, self.distance * 0.002)
        self.far = max(100.0, self.distance * 100.0)
        self.background = torch.zeros(3, dtype=torch.float32, device="cuda")
        self.projection = getProjectionMatrix(self.near, self.far, self.fov_x, self.fov_y).cuda().T
        self.pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)

    def render_frame(self, yaw, pitch, zoom, pan_x, pan_y, quality="preview"):
        horizon = math.cos(yaw) * self.outward + math.sin(yaw) * self.right
        outward = math.cos(pitch) * horizon + math.sin(pitch) * self.up
        right = unit_vector(np.cross(self.up, outward))
        forward = -outward
        down = np.cross(forward, right)
        distance = self.distance / zoom
        position = self.focus + distance * (outward + pan_x * right - pan_y * down)
        world_to_camera = np.eye(4, dtype=np.float32)
        world_to_camera[:3, :3] = np.stack((right, down, forward))
        world_to_camera[:3, 3] = -world_to_camera[:3, :3] @ position

        width, height = self.dimensions[quality]
        high = quality == "high"
        scale = HIGH_QUALITY_SUPERSAMPLING if high else 1
        with self.torch.no_grad():
            view_transform = self.torch.as_tensor(world_to_camera, device="cuda").T
            camera = self.make_camera(
                round(width * scale), round(height * scale), self.fov_y, self.fov_x, self.near, self.far,
                view_transform, view_transform @ self.projection,
            )
            frame = self.render(camera, self.gaussians, self.pipe, self.background)["render"]
            pixels = frame.clamp_(0, 1).mul_(255).round_().to(self.torch.uint8)
            image = Image.fromarray(pixels.permute(1, 2, 0).contiguous().cpu().numpy())

        with io.BytesIO() as buffer:
            if high:
                image = image.resize((width, height), Image.Resampling.LANCZOS)
                image.save(buffer, format="JPEG", quality=94, subsampling=0)
            else:
                image.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.gs_root.resolve()))

    # Keep setup diagnostics and native CUDA messages off the frame protocol.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1) as replies:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

        def reply(message):
            replies.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")

        try:
            renderer = InteractiveRenderer(args.model.resolve(), args.cameras.resolve())
        except Exception:
            traceback.print_exc()
            reply({"error": "交互渲染器初始化失败。请检查训练模型、相机文件和 CUDA 环境。"})
            return 1

        reply({"ready": True})
        for line in sys.stdin:
            try:
                pose = read_pose(line)
            except ValueError as exc:
                reply({"error": str(exc)})
                continue
            try:
                jpeg = renderer.render_frame(*pose)
            except Exception:
                traceback.print_exc()
                reply({"error": "当前相机视图渲染失败。"})
            else:
                reply({"jpeg": jpeg})
    return 0


if __name__ == "__main__":
    sys.exit(main())
