"""Render a small set of calibrated views from a trained 3DGS model."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image


MAX_IMAGE_SIZE = 800
VIEW_LIMIT = 3


@dataclass(frozen=True)
class View:
    name: str
    image: Path
    rotation: np.ndarray  # Camera-to-world with COLMAP's axis conventions.
    position: np.ndarray
    fov_x: float
    fov_y: float | None = None
    principal_point: tuple[float, float] = (0.5, 0.5)


def reference_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"相机图像路径应位于数据目录中：{name}")
    return root / relative


def colmap_views(source, calibrated_principal_point=False):
    from scene.colmap_loader import (
        qvec2rotmat,
        read_extrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_binary,
        read_intrinsics_text,
    )
    from utils.graphics_utils import focal2fov

    model = source / "sparse" / "0"
    if all((model / name).is_file() for name in ("images.bin", "cameras.bin")):
        images = read_extrinsics_binary(model / "images.bin")
        cameras = read_intrinsics_binary(model / "cameras.bin")
    else:
        images = read_extrinsics_text(model / "images.txt")
        cameras = read_intrinsics_text(model / "cameras.txt")

    views = []
    for image in images.values():
        camera = cameras[image.camera_id]
        if camera.model == "SIMPLE_PINHOLE":
            fx = fy = camera.params[0]
        elif camera.model == "PINHOLE":
            fx, fy = camera.params[:2]
        else:
            raise ValueError(f"相机 {image.name} 需要先去畸变（{camera.model}）。")
        principal_point = (0.5, 0.5)
        if calibrated_principal_point:
            cx, cy = camera.params[-2:]
            principal_point = (cx / camera.width, cy / camera.height)
        rotation = qvec2rotmat(image.qvec).T
        views.append(View(
            image.name,
            reference_path(source / "images", image.name),
            rotation,
            -rotation @ image.tvec,
            focal2fov(fx, camera.width),
            focal2fov(fy, camera.height),
            principal_point,
        ))
    return views


def blender_views(source):
    views = {}
    for split in ("train", "test"):
        contents = json.loads((source / f"transforms_{split}.json").read_text(encoding="utf-8"))
        for frame in contents["frames"]:
            name = frame["file_path"]
            image = reference_path(source, name)
            if not image.suffix:
                image = image.with_suffix(".png")
            transform = np.array(frame["transform_matrix"], dtype=np.float64)
            # Blender has Y up and Z back, while COLMAP has Y down and Z forward.
            transform[:3, 1:3] *= -1
            views[image] = View(
                image.relative_to(source).as_posix(),
                image,
                transform[:3, :3],
                transform[:3, 3],
                float(contents["camera_angle_x"]),
            )
    return list(views.values())


def choose_views(views):
    candidates = sorted((view for view in views if view.image.is_file()), key=lambda view: view.name)
    if not candidates:
        raise ValueError("数据目录中找不到已标定的参考图像。")
    if len(candidates) <= VIEW_LIMIT:
        return candidates

    positions = np.stack([view.position for view in candidates])
    forward = np.stack([view.rotation[:, 2] for view in candidates])
    scale = max(float(np.median(np.linalg.norm(positions - np.median(positions, axis=0), axis=1))), 1e-6)
    front_indices = [index for index, view in enumerate(candidates) if view.name.startswith("front-forward/")]
    selected = [front_indices[len(front_indices) * 4 // 5] if front_indices else len(candidates) // 5]
    closest = np.full(len(candidates), np.inf)
    while len(selected) < VIEW_LIMIT:
        index = selected[-1]
        position_distance = np.linalg.norm(positions - positions[index], axis=1) / scale
        direction_distance = np.linalg.norm(forward - forward[index], axis=1)
        closest = np.minimum(closest, position_distance + direction_distance * 0.5)
        closest[selected] = -1
        selected.append(int(np.argmax(closest)))
    return [candidates[index] for index in selected]


def load_reference(path):
    with Image.open(path) as image:
        image.draft("RGB", (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            black = Image.new("RGB", rgba.size)
            black.paste(rgba, mask=rgba.getchannel("A"))
            return black
        return image.convert("RGB")


def model_degree(path):
    from plyfile import PlyData

    properties = PlyData.read(path)["vertex"].properties
    higher_coefficients = sum(prop.name.startswith("f_rest_") for prop in properties)
    degree = math.isqrt(higher_coefficients // 3 + 1) - 1
    if not 0 <= degree <= 3 or 3 * ((degree + 1) ** 2 - 1) != higher_coefficients:
        raise ValueError("PLY 中的球谐系数数量与 3DGS 光栅化器不兼容。")
    return degree


def render_calibrated_views(views, model):
    import torch
    from gaussian_renderer import render
    from scene.cameras import MiniCam
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import focal2fov, fov2focal, getProjectionMatrix

    near, far = 0.01, 100.0
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    with torch.no_grad():
        gaussians = GaussianModel(model_degree(model))
        gaussians.load_ply(str(model))
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        for view in views:
            reference = load_reference(view.image)
            width, height = reference.size
            fov_y = view.fov_y if view.fov_y is not None else focal2fov(fov2focal(view.fov_x, width), height)
            world_to_camera = np.eye(4, dtype=np.float32)
            world_to_camera[:3, :3] = view.rotation.T
            world_to_camera[:3, 3] = -view.rotation.T @ view.position
            view_transform = torch.as_tensor(world_to_camera, device="cuda").T
            projection = getProjectionMatrix(near, far, view.fov_x, fov_y, view.principal_point).cuda().T
            camera = MiniCam(width, height, fov_y, view.fov_x, near, far, view_transform, view_transform @ projection)
            pixels = (render(camera, gaussians, pipe, background)["render"] * 255).round().to(torch.uint8)
            image = Image.fromarray(pixels.permute(1, 2, 0).contiguous().cpu().numpy())
            yield view, image, reference


def render_previews(source, model, output, calibrated_principal_point=False, held_out=False):
    candidates = (colmap_views(source, calibrated_principal_point) if (source / "sparse").exists()
                  else blender_views(source))
    if held_out:
        test_file = source / "sparse" / "0" / "test.txt"
        test_names = set(test_file.read_text(encoding="utf-8").splitlines())
        candidates = [view for view in candidates if view.name in test_names]
    views = choose_views(candidates)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    with tempfile.TemporaryDirectory(prefix=".render-", dir=output) as staging:
        staged = Path(staging)
        for index, (view, image, reference) in enumerate(render_calibrated_views(views, model), 1):
            render_name = f"view-{index:02d}-render.jpg"
            reference_name = f"view-{index:02d}-reference.jpg"
            image.save(staged / render_name, quality=91)
            reference.save(staged / reference_name, quality=91)
            manifest.append({"name": view.name, "render": render_name, "reference": reference_name})

        (staged / "views.json").write_text(json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8")
        for item in manifest:
            for key in ("render", "reference"):
                (staged / item[key]).replace(output / item[key])
        (staged / "views.json").replace(output / "views.json")


def render_timeline(source, model, output):
    views = sorted(colmap_views(source), key=lambda view: view.image.stem)
    timestamps = [view.image.stem for view in views]
    if (not timestamps or len(set(timestamps)) != len(timestamps)
            or any(len(stamp) != 16 or not stamp.isascii() or not stamp.isdecimal() for stamp in timestamps)):
        raise ValueError("时间轴需要一路带有微秒时间戳的 COLMAP 相机图像。")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".timeline-", dir=output.parent) as temporary:
        staged = Path(temporary) / "frames"
        staged.mkdir()
        for index, (_, image, reference) in enumerate(render_calibrated_views(views, model)):
            image.save(staged / f"{index:04d}-render.jpg", quality=87)
            reference.save(staged / f"{index:04d}-reference.jpg", quality=87)
        (staged / "timestamps.json").write_text(json.dumps(timestamps) + "\n", encoding="utf-8")
        staged.rename(output)
    print(f"已生成 {len(timestamps)} 组相机帧。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeline", action="store_true", help="Render all timestamped COLMAP views in time order.")
    parser.add_argument("--calibrated-principal-point", action="store_true")
    parser.add_argument("--held-out", action="store_true", help="Choose preview images from sparse/0/test.txt.")
    args = parser.parse_args()
    sys.path.insert(0, str(args.gs_root.resolve()))
    source, model, output = args.source.resolve(), args.model.resolve(), args.output.resolve()
    if args.timeline:
        render_timeline(source, model, output)
    else:
        render_previews(source, model, output, args.calibrated_principal_point, args.held_out)


if __name__ == "__main__":
    main()
