"""Prepare COLMAP data and launch the local 3D Gaussian Splatting trainer."""

import csv
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys


PROGRESS_PATTERN = re.compile(r"Training progress:[^\r\n]*?\|\s*(\d+)\s*/\s*\d+")
SAVING_PATTERN = re.compile(r"\[ITER (\d+)\] Saving Gaussians")


def detect_gpus():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    gpus = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 4:
            continue
        index, name, uuid, memory = (value.strip() for value in row)
        if not index.isdecimal() or not uuid.startswith("GPU-"):
            continue
        gpus.append({
            "index": int(index),
            "name": name,
            "uuid": uuid,
            "memoryMiB": int(memory) if memory.isdecimal() else None,
        })
    return sorted(gpus, key=lambda gpu: gpu["index"])


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
    requested_gpu = payload.get("gpu")
    if requested_gpu in (None, "", "auto"):
        gpu = next((card for card in gpus if "RTX 5090" in card["name"].upper()), gpus[0])
    else:
        gpu = next((card for card in gpus if str(requested_gpu) in (str(card["index"]), card["uuid"])), None)
        if gpu is None:
            raise ValueError("所选 NVIDIA GPU 不可用。")

    environment = dict(os.environ)
    environment.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpu["uuid"],
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    probe_command = [gs_python, "-u", str(Path(__file__).resolve()), "--probe", str(root)]

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
        environment=environment,
        dataset=dataset,
        probe_command=probe_command,
        command=command,
        undistort_command=undistort_command,
    )


def run_training(plan, execute):
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
    except ImportError as exc:
        raise RuntimeError(f"训练 Python 缺少 PyTorch：{exc}") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无法使用所选 NVIDIA GPU。请检查训练 Python 的 CUDA 版 PyTorch 和驱动。")

    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    print(f"训练 GPU：{name}（sm_{major}{minor}），PyTorch {torch.__version__}，CUDA {torch.version.cuda}。")
    try:
        importlib.import_module("train")
        from diff_gaussian_rasterization import _C as rasterizer
        from simple_knn._C import distCUDA2
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"3DGS 依赖加载失败：{exc}。请检查训练环境中的 Python 包和 CUDA 扩展。") from exc

    try:
        positions = torch.tensor(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.1], [0.0, 0.1, 1.2], [-0.1, 0.0, 1.3]],
            device=device,
        )
        identity = torch.eye(4, device=device)
        distCUDA2(positions)
        rasterizer.mark_visible(positions, identity, identity)
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        raise RuntimeError(
            f"3DGS 的 PyTorch 或 CUDA 扩展无法在 sm_{major}{minor} 上执行：{exc}。"
            "请使用支持该显卡的 PyTorch，并为此架构重新编译 CUDA 扩展。"
        ) from exc
    print("PyTorch、simple-knn 和 3DGS 光栅化 CUDA 检查通过。")


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--probe":
    try:
        probe_training_environment(sys.argv[2])
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
