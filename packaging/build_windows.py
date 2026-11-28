"""Build the 3Dmap Windows x64 portable edition from the working runtime."""

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent
VGGT_REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
CUDA_CAPABILITIES = (75, 80, 86, 89, 120)
CUDA_ARCH_LIST = " ".join(f"{arch // 10}.{arch % 10}" for arch in CUDA_CAPABILITIES) + "+PTX"
RUNTIME_PACKAGES = (
    "torch", "torchvision", "numpy", "Pillow", "opencv-python-headless",
    "pycolmap", "plyfile", "tqdm", "simple-knn", "diff-gaussian-rasterization",
    "huggingface-hub", "safetensors", "einops",
)
WEB_FILES = (
    "serve.py", "photo_jobs.py", "training.py", "dataset_timeline.py",
    "interactive_process.py", "render_preview.py", "render_interactive.py",
    "vggt_reconstruction.py", "studio.html", "studio.css", "studio.js",
    "interactive-view.js",
)
README = """3Dmap · Windows 便携版
=====================

双击 3Dmap.exe 打开工作台。系统托盘里的 3Dmap 菜单可以再次打开界面或退出。
可在首页体验内置的庭院花簇示例。自由视角中按住滚轮拖动旋转，滚动缩放，
右键拖动平移。双击画面可重置角度。

生成自己的模型时，选择 VGGT 或 COLMAP，导入重叠的 JPG、PNG 或 WebP，
设置训练步数，点击「开始生成」。3DGS 完成后可查看渲染和下载 PLY。

本包适用于 Windows x64，包含 Python、CUDA 运行依赖和 VGGT 本地权重。
目前覆盖 NVIDIA GTX 16 系和 RTX 20、30、40、50 系，范围指 3DGS 训练和自由视角的 CUDA 构建。
GTX 10 系及更早型号、AMD 和 Intel 显卡不在本次覆盖范围内。
构建覆盖范围不代表每个型号均已完成实机验证。
打包时会检查 PyTorch 和 3DGS 扩展是否覆盖计算能力 7.5、8.0、8.6、8.9 和 12.0。
训练和自由视角需要上述 NVIDIA 显卡、兼容打包 CUDA 版本的驱动和足够的显存。
预览图片可在无 GPU 时查看。

可将整个文件夹复制到本机其他位置。工作台仅监听本机回环地址。
生成记录与日志保存在当前用户的 %LOCALAPPDATA%\\3Dmap 下。
第三方许可随组件附带，示例模型来源见 web/sample/ATTRIBUTION.txt。
"""


def ignore_generated(_directory, names):
    return [name for name in names if name in {"__pycache__", ".git", ".idea", ".gitignore"}
            or name.endswith((".pyc", ".pdb"))]


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source, target):
    shutil.copytree(source, target, ignore=ignore_generated)


def installed_distributions(names):
    pending = list(names)
    installed = {}
    while pending:
        name = pending.pop()
        key = canonicalize_name(name)
        if key in installed:
            continue
        distribution = metadata.distribution(name)
        installed[key] = distribution
        for declared in distribution.requires or ():
            requirement = Requirement(declared)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return installed


def runtime_contents(destination):
    python_home = Path(sys.base_prefix)
    for item in python_home.iterdir():
        if item.suffix.lower() == ".dll" or item.name in {"python.exe", "LICENSE.txt"}:
            if item.is_file():
                copy_file(item, destination / item.name)
    copy_tree(python_home / "DLLs", destination / "DLLs")

    standard = python_home / "Lib"
    lib = destination / "Lib"
    lib.mkdir()
    for source in standard.iterdir():
        if source.name in {"site-packages", "test", "idlelib", "tkinter", "turtledemo", "venv", "ensurepip"}:
            continue
        if source.is_dir():
            if not ignore_generated(standard, [source.name]):
                copy_tree(source, lib / source.name)
        elif source.is_file():
            copy_file(source, lib / source.name)

    site = lib / "site-packages"
    packages = installed_distributions(RUNTIME_PACKAGES)
    for name, distribution in sorted(packages.items()):
        directory = Path(distribution.locate_file("")).resolve()
        for item in distribution.files or ():
            source = Path(distribution.locate_file(item)).resolve()
            if not source.is_relative_to(directory) or not source.is_file():
                continue
            relative = source.relative_to(directory)
            if "__pycache__" in relative.parts or source.suffix in {".pyc", ".pdb", ".lib"}:
                continue
            if name == "torch" and relative.parts[:2] in {("torch", "include"), ("torch", "share")}:
                continue
            copy_file(source, site / relative)
        print(f"Bundled {name} {distribution.version}", flush=True)

    python_tag = f"{sys.version_info.major}{sys.version_info.minor}"
    (destination / f"python{python_tag}._pth").write_text(
        ".\nDLLs\nLib\nLib/site-packages\n../web\n../gaussian-splatting/gaussian-splatting\nimport site\n",
        encoding="ascii",
    )
    return packages


def format_architectures(architectures):
    return ", ".join(f"{arch // 10}.{arch % 10}" for arch in architectures)


def require_cuda_coverage(component, architectures):
    # A native cubin can run on later minor revisions within the same major revision.
    missing = [
        target for target in CUDA_CAPABILITIES
        if not any(arch // 10 == target // 10 and arch <= target for arch in architectures)
    ]
    if missing:
        raise RuntimeError(f"{component} 缺少计算能力 {format_architectures(missing)} 的 CUDA 二进制代码。")


def find_cuobjdump(explicit):
    if explicit:
        return shutil.which(explicit)
    if toolkit := os.environ.get("CUDA_PATH"):
        executable = Path(toolkit) / "bin" / "cuobjdump.exe"
        if executable.is_file():
            return str(executable)
    return shutil.which("cuobjdump.exe")


def verify_cuda_runtime(runtime, cuobjdump):
    # Read the compiled list even on build hosts without a visible CUDA device.
    probe = """
import json
import torch

if not torch.version.cuda:
    raise SystemExit("便携版需要支持目标显卡的 CUDA 版 PyTorch。")

import torchvision
import pycolmap
from diff_gaussian_rasterization import _C as rasterizer
from simple_knn import _C as knn

get_flags = getattr(torch._C, "_cuda_getArchFlags", None)
architectures = get_flags() if get_flags else None
if not architectures:
    raise SystemExit("无法读取当前 PyTorch 编译的 CUDA 架构列表。")
print(json.dumps({"cuda": torch.version.cuda, "architectures": architectures,
                  "extensions": {"simple-knn": knn.__file__,
                                 "diff-gaussian-rasterization": rasterizer.__file__}}))
"""
    result = subprocess.run(
        [str(runtime / "python.exe"), "-B", "-X", "utf8", "-c", probe],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    if result.returncode:
        raise RuntimeError(f"打包后的 Python 无法加载 CUDA 或重建依赖。\n{result.stderr.strip()}")
    info = json.loads(result.stdout)
    torch_arches = {int(arch) for arch in re.findall(r"\bsm_(\d+)\b", info["architectures"])}
    require_cuda_coverage("PyTorch", torch_arches)
    for component, binary in info["extensions"].items():
        path = Path(binary).resolve()
        if not path.is_relative_to(runtime) or path.suffix.lower() != ".pyd":
            raise RuntimeError(f"{component} 未从便携版加载 Windows 扩展：{path}")
        result = subprocess.run(
            [cuobjdump, "--list-elf", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
        if result.returncode:
            raise RuntimeError(f"无法检查 {component} 的 CUDA 二进制代码。\n{result.stderr.strip()}")
        arches = {int(arch) for arch in re.findall(r"\.sm_(\d+)\.cubin\b", result.stdout)}
        try:
            require_cuda_coverage(component, arches)
        except RuntimeError as error:
            raise RuntimeError(f"{error} 请设置 TORCH_CUDA_ARCH_LIST='{CUDA_ARCH_LIST}' 后重新编译。") from error
    print(f"Bundled CUDA {info['cuda']} supports compute capabilities {format_architectures(CUDA_CAPABILITIES)}",
          flush=True)
    return info["cuda"]


def copy_studio(destination):
    web = ROOT / "web"
    for name in WEB_FILES:
        copy_file(web / name, destination / "web" / name)
    copy_tree(web / "sample", destination / "web" / "sample")

    gs = ROOT / "gaussian-splatting" / "gaussian-splatting"
    gs_target = destination / "gaussian-splatting" / "gaussian-splatting"
    for directory in ("arguments", "gaussian_renderer", "scene", "utils"):
        copy_tree(gs / directory, gs_target / directory)
    for name in ("train.py", "LICENSE.md"):
        copy_file(gs / name, gs_target / name)

    license_folder = destination / "Licenses"
    for component in ("simple-knn", "diff-gaussian-rasterization"):
        for item in (gs / "submodules" / component).glob("LICENSE*"):
            copy_file(item, license_folder / f"{component}-{item.name}")

    cache = web / ".vggt"
    source = cache / f"vggt-{VGGT_REVISION}"
    target = destination / "web" / ".vggt" / source.name
    copy_tree(source / "vggt", target / "vggt")
    copy_file(source / "LICENSE.txt", target / "LICENSE.txt")
    return cache / "model.safetensors", target.parent / "model.safetensors"


def save_vggt_weights(source, target):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    # The studio creates VGGT with the tracking and point heads disabled.
    # Its aggregator is converted to BF16 before inference on the supported GPUs.
    state = {}
    with safe_open(source, framework="pt", device="cpu") as model:
        for key in model.keys():
            if key.startswith(("aggregator.", "camera_head.", "depth_head.")):
                tensor = model.get_tensor(key)
                state[key] = tensor.to(torch.bfloat16) if key.startswith("aggregator.") else tensor
        save_file(state, target, metadata={
            "source": "https://huggingface.co/facebook/VGGT-1B",
            "license": "CC-BY-NC-4.0",
            "changes": "Unused point and track heads omitted. Aggregator weights stored in BF16.",
        })
    print(f"Bundled VGGT weights: {target.stat().st_size:,} bytes", flush=True)


def build_launcher(destination):
    compiler = Path("C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe")
    source = ROOT / "packaging" / "Launcher.cs"
    icon = destination / "3Dmap.ico"
    image = Image.new("RGBA", (128, 128))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((4, 4, 124, 124), 29, fill="#3157d5")
    draw.rounded_rectangle((32, 30, 81, 81), 11, outline="#ffffff", width=5)
    draw.rounded_rectangle((46, 45, 95, 96), 11, outline="#bdcdff", width=5)
    image.save(icon, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128)])
    subprocess.run([
        str(compiler), "/nologo", "/target:winexe", "/platform:x64", "/codepage:65001",
        "/reference:System.Windows.Forms.dll", "/reference:System.Drawing.dll", f"/win32icon:{icon}",
        f"/out:{destination / '3Dmap.exe'}", str(source),
    ], check=True)
    icon.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "3Dmap-Windows-x64")
    parser.add_argument("--cuobjdump", help="Path to CUDA Toolkit's cuobjdump.exe (default: CUDA_PATH or PATH)")
    args = parser.parse_args()
    if sysconfig.get_platform() != "win-amd64":
        parser.error("Build the Windows x64 portable edition with x64 Python on Windows.")
    cuobjdump = find_cuobjdump(args.cuobjdump)
    if cuobjdump is None:
        parser.error("Cannot find CUDA Toolkit's cuobjdump.exe. Set CUDA_PATH or pass --cuobjdump.")
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".3Dmap-build-", dir=target.parent) as temporary:
        staging = Path(temporary)
        print(f"Building in {staging}", flush=True)
        build_launcher(staging)
        runtime = staging / "runtime"
        packages = runtime_contents(runtime)
        try:
            cuda_version = verify_cuda_runtime(runtime, cuobjdump)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            parser.exit(1, f"CUDA runtime validation failed: {error}\n")
        weights, destination = copy_studio(staging)
        save_vggt_weights(weights, destination)
        (staging / "Start here.txt").write_text(README, encoding="utf-8-sig")
        (staging / "Licenses").mkdir(exist_ok=True)
        (staging / "Licenses" / "Runtime components.txt").write_text(
            f"CUDA (PyTorch): {cuda_version}\n"
            f"CUDA compute capabilities: {format_architectures(CUDA_CAPABILITIES)}\n"
            + "\n".join(f"{name} {distribution.version}" for name, distribution in sorted(packages.items())) + "\n",
            encoding="utf-8",
        )
        (staging / "Licenses" / "VGGT weights.txt").write_text(
            "Derived from the official Meta VGGT-1B model.\nhttps://huggingface.co/facebook/VGGT-1B\n"
            "CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/\n"
            "This package removes the unused heads and stores aggregator weights in BF16.\n",
            encoding="utf-8",
        )
        staging.rename(target)
    size = sum(item.stat().st_size for item in target.rglob("*") if item.is_file())
    print(f"Portable edition ready: {target} ({size / 1024**3:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
