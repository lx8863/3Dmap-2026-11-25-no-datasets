"""Prepare COLMAP data and launch the local 3D Gaussian Splatting trainer."""

import csv
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace


PROGRESS_PATTERN = re.compile(r"Training progress:[^\r\n]*?\|\s*(\d+)\s*/\s*\d+")
SAVING_PATTERN = re.compile(r"\[ITER (\d+)\] Saving Gaussians")


def detect_gpus():
    fields = "index,name,uuid,memory.total"
    for query in (f"{fields},compute_cap", fields):
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.CalledProcessError:
            # Older versions of nvidia-smi may not expose compute_cap.
            continue
        except (OSError, subprocess.TimeoutExpired):
            return []
        break
    else:
        return []

    gpus = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) not in (4, 5):
            continue
        index, name, uuid, memory, *capabilities = (value.strip() for value in row)
        if not index.isdecimal() or not uuid.startswith("GPU-"):
            continue
        capability = capabilities[0] if capabilities else ""
        gpus.append({
            "index": int(index),
            "name": name,
            "uuid": uuid,
            "memoryMiB": int(memory) if memory.isdecimal() else None,
            "computeCapability": capability if re.fullmatch(r"[0-9]+\.[0-9]+", capability) else None,
        })
    return sorted(gpus, key=lambda gpu: gpu["index"])


def select_gpu(gpus, requested_gpu=None):
    if requested_gpu in (None, "", "auto"):
        return max(gpus, key=lambda card: card["memoryMiB"] or 0, default=None)
    return next((card for card in gpus if str(requested_gpu) in (str(card["index"]), card["uuid"])), None)


def cuda_environment(gpu):
    return {
        **os.environ,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpu["uuid"],
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }


def training_probe_command(gs_python, gs_root):
    return [gs_python, "-u", str(Path(__file__).resolve()), "--probe", str(gs_root)]


def training_progress(line):
    match = PROGRESS_PATTERN.search(line) or SAVING_PATTERN.search(line)
    return int(match[1]) if match else None


def training_path(value, label):
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"请输入{label}。")
    try:
        return Path(value.strip()).expanduser().resolve()
    except RuntimeError as exc:
        raise ValueError(f"{label}的路径无效。") from exc


def check_output_directory(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("训练输出目录必须是新目录或空目录，以保留已有训练结果。")


def has_sparse_model(path, allow_ply=False):
    cameras_and_images = any(
        all((path / f"{name}.{extension}").is_file() for name in ("cameras", "images"))
        for extension in ("bin", "txt")
    )
    point_extensions = ("bin", "txt", "ply") if allow_ply else ("bin", "txt")
    return cameras_and_images and any(
        (path / f"points3D.{extension}").is_file() for extension in point_extensions
    )


@dataclass
class TrainingPlan:
    output_path: Path
    total: int
    gpu: dict
    root: Path
    environment: dict
    dataset: Path
    probe_command: list
    command: list
    undistort_command: list | None = None


def make_training_plan(payload, *, gs_python, gs_root, colmap, gpus):
    if not isinstance(payload, dict):
        raise ValueError("训练设置格式无效。")
    if not gs_python or not (Path(gs_root) / "train.py").is_file():
        raise ValueError("未找到 3DGS 训练程序。请指定训练 Python 和仓库目录。")

    root = Path(gs_root).resolve()
    source_type = payload.get("sourceType")
    if source_type not in ("dataset", "colmap"):
        raise ValueError("请选择训练数据来源。")
    source = training_path(payload.get("sourcePath"), "训练数据目录")
    if not source.is_dir():
        raise ValueError(f"训练数据目录 {source} 不存在。")
    output = training_path(payload.get("modelPath"), "训练输出目录")
    protected = (source, source / "images", source / "sparse")
    if output in protected or any(path in output.parents for path in protected[1:]):
        raise ValueError("请把模型输出放在训练数据的图像和稀疏模型目录以外。")
    check_output_directory(output)

    total = payload.get("iterations")
    if type(total) is not int or total < 1:
        raise ValueError("训练迭代次数必须是正整数。")
    if not gpus:
        raise ValueError("未检测到可用于 3DGS 训练的 NVIDIA GPU。")
    gpu = select_gpu(gpus, payload.get("gpu"))
    if gpu is None:
        raise ValueError("所选 NVIDIA GPU 不可用。")

    dataset = source
    undistort_command = None
    if source_type == "dataset":
        colmap_dataset = source / "sparse" / "0"
        if (source / "sparse").exists():
            if not (source / "images").is_dir() or not has_sparse_model(colmap_dataset, allow_ply=True):
                raise ValueError("3DGS 数据需要 images/ 和包含相机、位姿、三维点的 sparse/0/。")
        elif not all((source / f"transforms_{split}.json").is_file() for split in ("train", "test")):
            raise ValueError("3DGS 数据目录需要 images/ 与 sparse/0/，或 Blender 的训练和测试 transforms 文件。")
    else:
        if not colmap:
            raise ValueError("准备 COLMAP 训练数据需要本地 COLMAP 可执行文件。")
        images = training_path(payload.get("imagePath"), "原始图像目录")
        if not images.is_dir():
            raise ValueError(f"原始图像目录 {images} 不存在。")
        if output == images or images in output.parents:
            raise ValueError("请把模型输出放在原始图像目录以外。")
        model = next((
            path for path in (source / "sparse" / "0", source / "sparse", source)
            if has_sparse_model(path)
        ), None)
        if model is None:
            raise ValueError("COLMAP 工作区内找不到可训练的稀疏模型。")
        dataset = output / "training_data"
        undistort_command = [
            colmap, "image_undistorter",
            "--image_path", str(images),
            "--input_path", str(model),
            "--output_path", str(dataset),
            "--output_type", "COLMAP",
        ]

    command = [
        gs_python, "-u", str(root / "train.py"),
        "--source_path", str(dataset),
        "--model_path", str(output),
        "--iterations", str(total),
        "--disable_viewer",
    ]
    return TrainingPlan(
        output_path=output,
        total=total,
        gpu=gpu,
        root=root,
        environment=cuda_environment(gpu),
        dataset=dataset,
        probe_command=training_probe_command(gs_python, root),
        command=command,
        undistort_command=undistort_command,
    )


def run_training(plan, execute, *, probe=True):
    if probe:
        execute("prepare", plan.probe_command, plan.root, plan.environment)
    plan.output_path.mkdir(parents=True, exist_ok=True)
    check_output_directory(plan.output_path)
    if plan.undistort_command:
        execute("prepare", plan.undistort_command, plan.root, plan.environment)
        sparse = plan.dataset / "sparse"
        model = sparse / "0"
        if not model.exists() and has_sparse_model(sparse):
            model.mkdir()
            for extension in ("bin", "txt"):
                for name in ("rigs", "cameras", "frames", "images", "points3D"):
                    model_file = sparse / f"{name}.{extension}"
                    if model_file.is_file():
                        model_file.rename(model / model_file.name)
        if not (plan.dataset / "images").is_dir() or not has_sparse_model(model):
            raise RuntimeError("COLMAP 去畸变未生成 3DGS 所需的训练图像和 sparse/0 模型。")
    execute("training", plan.command, plan.root, plan.environment)


def probe_training_environment(root):
    """Run lightweight CUDA checks inside the selected 3DGS Python environment."""
    sys.path.insert(0, root)
    try:
        import torch
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(f"无法加载 PyTorch：{exc}。请安装支持本机 GPU 的 CUDA 版 PyTorch。") from exc

    if torch.version.cuda is None:
        raise RuntimeError(f"当前 PyTorch {torch.__version__} 未包含 CUDA 支持。请安装 CUDA 版 PyTorch。")
    if not torch.cuda.is_available():
        raise RuntimeError(f"当前 PyTorch（CUDA {torch.version.cuda}）无法访问所选 NVIDIA GPU。"
                           "请确认显卡可见，且 NVIDIA 驱动支持这一 CUDA 版本。")

    device = torch.device("cuda:0")
    try:
        name = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
    except RuntimeError as exc:
        raise RuntimeError(f"无法初始化所选 NVIDIA GPU：{exc}。请检查 NVIDIA 驱动和所选显卡。") from exc
    architecture = f"sm_{major}{minor}"
    arch_list = ", ".join(torch.cuda.get_arch_list()) or "未报告"
    print(f"训练 GPU：{name}（{architecture}），PyTorch {torch.__version__}，CUDA {torch.version.cuda}。")
    print(f"PyTorch 编译目标：{arch_list}。")

    try:
        trainer = importlib.import_module("train")
        import numpy as np
        from gaussian_renderer import render
        from scene.cameras import MiniCam
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import BasicPointCloud, getProjectionMatrix
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(f"无法加载 {architecture} 的 3DGS 依赖：{exc}。"
                           "请检查 PyTorch、CUDA 工具链与 3DGS 扩展的版本是否匹配。") from exc

    component = "PyTorch"
    try:
        background = torch.zeros(3, device=device)
        camera_transform = torch.eye(4, device=device)
        projection = getProjectionMatrix(0.01, 10.0, 1.0, 1.0).to(device).T.contiguous()
        camera = MiniCam(16, 16, 1.0, 1.0, 0.01, 10.0, camera_transform, projection)
        torch.cuda.synchronize(device)

        component = "simple-knn"
        positions = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.1], [0.0, 0.1, 1.2], [-0.1, 0.0, 1.3]],
                             dtype=np.float32)
        point_cloud = BasicPointCloud(positions, np.ones_like(positions), np.zeros_like(positions))
        gaussians = GaussianModel(3)
        gaussians.create_from_pcd(point_cloud, [], 1.0)
        torch.cuda.synchronize(device)

        component = "diff-gaussian-rasterization"
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
        image = render(camera, gaussians, pipe, background, separate_sh=trainer.SPARSE_ADAM_AVAILABLE)["render"]
        image.mean().backward()
        torch.cuda.synchronize(device)

        if trainer.FUSED_SSIM_AVAILABLE:
            component = "fused-ssim"
            rendered = image.detach().unsqueeze(0).requires_grad_()
            trainer.fused_ssim(rendered, torch.zeros_like(rendered)).backward()
            torch.cuda.synchronize(device)
    except (OSError, RuntimeError, TypeError) as exc:
        if component == "PyTorch":
            hint = f"请安装支持 {architecture} 的 PyTorch，并确认 NVIDIA 驱动兼容。"
        else:
            hint = f"请在当前 PyTorch 和 CUDA 环境中为 {architecture} 重新编译 {component}。"
        if component == "fused-ssim":
            hint += "也可以移除该可选扩展以使用 PyTorch SSIM。"
        raise RuntimeError(f"{name}（{architecture}）上的 {component} CUDA 检查失败：{exc}。{hint}") from exc
    print("PyTorch、simple-knn 和 3DGS 光栅化前向及反向 CUDA 检查通过。")


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--probe":
    try:
        probe_training_environment(sys.argv[2])
    except RuntimeError as exc:
        print(f"3DGS: {exc}", file=sys.stderr)
        sys.exit(1)
