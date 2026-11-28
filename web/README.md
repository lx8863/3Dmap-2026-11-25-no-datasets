# 3Dmap 本地工作台

Windows 双击仓库根目录的 `Start-3Dmap.cmd`，浏览器会自动打开工作台。也可以在仓库根目录运行 `python -B web/serve.py --open`，手动启动的默认地址是 <http://127.0.0.1:8765/>。

## 从照片生成 3D 模型

1. 在「标定方式」选 VGGT 或 COLMAP。VGGT 支持 2–12 张照片，COLMAP 支持 3–128 张照片。
2. 点击「选择照片」在系统文件选择器中多选 JPG、PNG 或 WebP，或直接拖入照片。可在页面移除照片。
3. 设定 3DGS 训练步数并点击「开始生成」。工作台自动完成相机标定、训练和渲染。
4. 在右侧切换渲染与原照片，或打开「自由视角」。点击「下载 PLY」保存 3DGS 训练生成的模型。支持系统保存对话框的浏览器会弹出文件位置选择。

照片和输出保存在运行工作台的机器上。任务目录位于 `outputs/workbench/`。默认首页会加载最近一次成功生成的模型。「高级工作台」保留本地目录、训练和时间轴的独立操作。

## 阅览高级工作台

访问 <http://127.0.0.1:8765/workbench>。可以从已有的训练数据开始，或单独运行 COLMAP、VGGT 和 3DGS。

## 局域网访问

在本机通过 Wi-Fi 提供访问时，用启动参数指定该网卡的 IPv4 地址和前缀。

```bash
python3 -B web/serve.py --lan 192.168.31.30/24
```

同一 Wi-Fi 子网中的设备可打开 <http://192.168.31.30:8765/>，用 `colmap` 和本机 `~/.config/colmap-workbench/lan-password` 文件里的口令登录。第一次启用局域网访问时会生成该文件。局域网和本机地址共用任务进度。图像目录和输出目录填写运行工作台的电脑上的路径。

## 从照片开始

1. 在「照片重建」选择 COLMAP，再填写原始图像目录和输出工作区的绝对路径，然后开始重建。
2. 在「3DGS 训练」选择「COLMAP 稀疏重建结果」，点击「使用最近的重建」。工作台会使用原始图像和稀疏模型准备去畸变数据。
3. 填写一个新建或空的模型输出目录，确认显卡和迭代数，开始训练。训练完成后，模型保存到输出目录下的 `point_cloud/iteration_<N>/point_cloud.ply`。
4. 完成后点「自由 3D」，拖动环绕、滚轮缩放，右键或按住 Shift 拖动以平移。用「相机视角」可查看按拍摄位姿生成的渲染并对照原图。点云视图可切换主体和全景。

已有去畸变数据时，选择「已准备的 3DGS 数据集」，填入包含 `images/` 和 `sparse/0/` 的目录即可。

## VGGT 少图重建（测试）

在「照片重建」把重建引擎切换到 **VGGT 少图重建（测试）**。图像目录放 2–12 张有视角重叠的 JPG、PNG 或 WebP 照片。工作区填写一个尚未创建的目录，点击「开始 VGGT 重建」。显存有限时，可以先用 2–4 张照片验证。

成功后可以旋转预览点云。导出的 `images/` 与 `sparse/0/` 构成 3DGS 数据集。在「3DGS 训练」选择「已准备的 3DGS 数据集」，点击「使用最近的重建」填入结果目录。

首次配置需要 NVIDIA GPU、支持该 GPU 的 CUDA 版 PyTorch 和 torchvision。以下命令在仓库根目录执行，虚拟环境复用当前 Python 的 CUDA PyTorch。

```powershell
python -m venv --system-site-packages web/.vggt/venv
web/.vggt/venv/Scripts/python.exe -m pip install "pycolmap==4.2.1" huggingface_hub safetensors einops
```

下载 [VGGT 源码](https://codeload.github.com/facebookresearch/vggt/zip/a288dd0f14786c93483e45524328726ab7b1b4ce) 并解压到 `web/.vggt/`，使该目录下包含 `vggt-a288dd0f14786c93483e45524328726ab7b1b4ce/`。把 [VGGT-1B 权重](https://huggingface.co/facebook/VGGT-1B/blob/main/model.safetensors) 保存为 `web/.vggt/model.safetensors`。默认权重约 5 GB，按模型卡的 CC BY-NC 4.0 许可使用。

然后运行 `python -B web/serve.py`。Linux 的虚拟环境 Python 位于 `web/.vggt/venv/bin/python`。已有独立环境时，可以用 `--vggt-python`、`--vggt-root`、`--vggt-weights` 指定解释器、VGGT 源码目录和权重文件。

## 运行环境

- 产品首页的 COLMAP 通过 PyCOLMAP 运行。在同一环境安装 `pycolmap==4.2.1`、Pillow、plyfile、tqdm 和 opencv-python-headless。
- 3DGS 环境需要可用的 CUDA PyTorch，以及为当前显卡编译的 `simple-knn` 和 `diff-gaussian-rasterization`。Windows 会优先使用 `web/.venv/Scripts/python.exe`，也可以用 `--gs-python` 指定环境。
- 服务默认监听 `127.0.0.1`，一次运行一个任务。使用 `--lan` 时，指定的网卡和子网也可访问。
- `serve.py` 会识别项目中已构建的 COLMAP、`web/.colmap-env/bin/colmap`、`PATH` 中的 `colmap` 和 `web/.colmap-apt/colmap`。用 `--colmap` 可以指定其他可执行文件。
- 3DGS 默认使用 `web/.venv/bin/python`。用 `--gs-python` 可以指定其他 Python 环境，用 `--gs-root` 可以指定包含 `train.py` 的 3DGS 源码目录。
- 重启服务时可附加 `--restore-training <模型输出目录>`，打开已保存的训练结果和渲染预览。
- 训练前会在所选显卡上执行 PyTorch、近邻计算和 3DGS 光栅化的 CUDA 检查。自动模式优先选择显存总量较大的显卡。高级工作台可指定显卡。
- 打开高级工作台 `index.html` 可以直接操作示例点云并查看 COLMAP 命令。启动本地服务后可以运行任务和读取实际模型。

## 显卡兼容与编译

照片标定可选 CPU 版 COLMAP。3DGS 训练、自由视角和 VGGT 需要支持当前显卡的 NVIDIA 驱动与 CUDA 版 PyTorch。VGGT 在计算能力 8.0 及以上使用 BF16，在 7.5 使用 FP16。显存较小的显卡可先用 COLMAP 标定少量照片。 [NVIDIA 显卡计算能力列表](https://developer.nvidia.com/cuda/gpus)

| 显卡 | CUDA 计算能力 |
| --- | --- |
| GeForce GTX 16 系、RTX 20 系 | 7.5 |
| GeForce RTX 30 系 | 8.6 |
| GeForce RTX 40 系 | 8.9 |
| GeForce RTX 50 系 | 12.0 |

从源码运行时，为目标显卡重编译 `simple-knn` 和 `diff-gaussian-rasterization`。构建机需要匹配 CUDA 版 PyTorch 的 CUDA Toolkit 和 C++ 编译器。为便携版覆盖上表所有显卡时，选用能编译 12.0 的 Toolkit，例如 CUDA 12.8 或 CUDA 13.x。 [PyTorch CUDA 扩展文档](https://docs.pytorch.org/docs/main/cpp_extension.html)

在 Windows PowerShell 中，从仓库根目录执行以下命令。先在该 VGGT 虚拟环境准备 CUDA 版 PyTorch、torchvision 和构建依赖 `setuptools`、`wheel`、`ninja`。将 MSVC 编译器及 CUDA Toolkit 的 `bin` 目录加入构建环境。

```powershell
$env:TORCH_CUDA_ARCH_LIST = "7.5 8.0 8.6 8.9 12.0+PTX"
web/.vggt/venv/Scripts/python.exe -m pip install --force-reinstall --no-build-isolation --no-deps --no-cache-dir `
  ./gaussian-splatting/gaussian-splatting/submodules/simple-knn `
  ./gaussian-splatting/gaussian-splatting/submodules/diff-gaussian-rasterization
```

便携版打包会用 CUDA Toolkit 的 `cuobjdump` 验证扩展是否包含目标 GPU 的代码。Linux 源码运行时，在相同的训练 Python 环境中编译这两个扩展，并按实际显卡设定 `TORCH_CUDA_ARCH_LIST`。如果还安装了可选的 `fused-ssim`，也要为同一架构重新编译，或在该训练环境中使用工程内置的 SSIM 回退实现。
