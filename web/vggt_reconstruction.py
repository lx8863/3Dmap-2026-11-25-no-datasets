"""Run VGGT camera and depth inference and export a 3DGS dataset."""

import argparse
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory


VGGT_CACHE = Path(__file__).resolve().parent / ".vggt"
VGGT_REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGES = 12


def image_files(directory):
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_inputs(images, output):
    if not images.is_dir():
        raise ValueError(f"图像目录 {images} 不存在。")
    if output == images or images in output.parents or output.exists():
        raise ValueError("请指定图像目录以外、尚未创建的输出工作区。")
    files = image_files(images)
    if not 2 <= len(files) <= MAX_IMAGES:
        raise ValueError(f"VGGT 测试版需要 2–{MAX_IMAGES} 张 JPG、PNG 或 WebP 照片，当前找到 {len(files)} 张。")
    return files


def export_sparse(directory, files, bounds, extrinsics, intrinsics, depth, confidence, rgb):
    import numpy as np
    import pycolmap
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
    heights, widths = depth.shape[1:3]
    y, x = np.ogrid[:heights, :widths]
    reconstruction = pycolmap.Reconstruction()
    rng = np.random.default_rng(42)

    for index, (image, (left, top, right, bottom, width, height)) in enumerate(zip(files, bounds)):
        camera_id = index + 1
        focal_x, focal_y = intrinsics[index, 0, 0], intrinsics[index, 1, 1]
        cx, cy = intrinsics[index, 0, 2], intrinsics[index, 1, 2]
        scale = max(width, height) / widths
        params = np.array([focal_x, focal_y, cx - left, cy - top]) * scale
        rotation = extrinsics[index, :, :3]
        translation = extrinsics[index, :, 3]
        if not np.isfinite(params).all() or min(params[:2]) <= 0 or not np.isfinite(extrinsics[index]).all():
            raise ValueError(f"无法为 {image.name} 估计有效的相机参数。")

        camera = pycolmap.Camera(
            model="PINHOLE", width=int(width), height=int(height), params=params, camera_id=camera_id,
        )
        reconstruction.add_camera_with_trivial_rig(camera)
        frame = pycolmap.Image(name=image.name, camera_id=camera_id, image_id=camera_id)
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), translation)
        reconstruction.add_image_with_trivial_frame(frame, pose)

        inside = (x >= left) & (x < right) & (y >= top) & (y < bottom)
        scores = confidence[index]
        finite = np.isfinite(points[index]).all(axis=-1) & np.isfinite(scores)
        valid = inside & finite & (depth[index, ..., 0] > 0) & (scores > 1)
        if not valid.any():
            continue
        # Confidence ranges vary by scene. Keep the upper half of valid pixels in each view.
        cutoff = np.median(scores[valid])
        rows, columns = np.nonzero(valid & (scores >= cutoff))
        limit = 100_000 // len(files)
        if len(rows) > limit:
            selected = rng.choice(len(rows), size=limit, replace=False)
            rows, columns = rows[selected], columns[selected]
        for point, color in zip(points[index, rows, columns], rgb[index, rows, columns]):
            reconstruction.add_point3D(point, pycolmap.Track(), color)

    count = reconstruction.num_points3D()
    if not count:
        raise ValueError("VGGT 未生成可信点云。请试用视角重叠更多、细节清晰的照片。")
    directory.mkdir(parents=True)
    reconstruction.write(directory)
    print(f"VGGT 导出 {len(files)} 个相机和 {count:,} 个三维点。")


def reconstruct(args):
    sys.path.insert(0, str(args.vggt_root))
    import torch
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    files = validate_inputs(args.images, args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT 需要可用的 NVIDIA CUDA PyTorch 环境。")

    print(f"读取 VGGT 权重与 {len(files)} 张照片。")
    weights = load_file(args.weights)
    model = VGGT(enable_point=False, enable_track=False).eval()
    model.load_state_dict({name: weights[name] for name in model.state_dict()})
    del weights

    major, _ = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    model.aggregator.to(dtype=dtype)
    model.to("cuda")
    images, bounds = load_and_preprocess_images_square([str(file) for file in files], target_size=518)
    print("VGGT 正在估计相机位姿和深度。")
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        batch = images.unsqueeze(0).to("cuda")
        tokens, patch_start = model.aggregator(batch)
        pose = model.camera_head(tokens)[-1].float()
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose, batch.shape[-2:])
        depth, confidence = model.depth_head(tokens, batch, patch_start)
        predictions = [
            value[0].float().cpu().numpy() for value in (extrinsics, intrinsics, depth, confidence)
        ]
    del model, batch, tokens, pose, extrinsics, intrinsics, depth, confidence
    rgb = images.permute(0, 2, 3, 1).mul(255).round().byte().numpy()

    print("VGGT 正在导出点云和 3DGS 数据集。")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".vggt-", dir=args.output.parent) as temporary:
        dataset = Path(temporary) / "dataset"
        export_sparse(dataset / "sparse" / "0", files, bounds.numpy(), *predictions, rgb)
        output_images = dataset / "images"
        output_images.mkdir()
        for file in files:
            shutil.copy2(file, output_images / file.name)
        if args.output.exists():
            raise ValueError("输出工作区已存在。请选择新的目录。")
        dataset.rename(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vggt-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    try:
        reconstruct(parser.parse_args())
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            message = "VGGT 显存不足。请关闭占用 GPU 的程序或减少照片后重试。"
        parser.exit(1, f"VGGT: {message}\n")


if __name__ == "__main__":
    main()
