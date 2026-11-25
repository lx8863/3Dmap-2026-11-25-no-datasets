# COLMAP 与 3DGS 本地工作台

在 `3Dmap` 目录运行以下命令，然后访问 <http://127.0.0.1:8765/>。

```bash
python3 -B web/serve.py
```

## 局域网访问

在本机通过 Wi-Fi 提供访问时，用启动参数指定该网卡的 IPv4 地址和前缀。

```bash
python3 -B web/serve.py --lan 192.168.31.30/24
```

同一 Wi-Fi 子网中的设备可打开 <http://192.168.31.30:8765/>，用 `colmap` 和本机 `~/.config/colmap-workbench/lan-password` 文件里的口令登录。第一次启用局域网访问时会生成该文件。局域网和本机地址共用任务进度。图像目录和输出目录填写运行工作台的电脑上的路径。

## 从照片开始

1. 在「COLMAP 重建」填写原始图像目录和输出工作区的绝对路径，然后开始重建。
2. 在「3DGS 训练」选择「COLMAP 稀疏重建结果」，点击「使用最近的重建」。工作台会使用原始图像和稀疏模型准备去畸变数据。
3. 填写一个新建或空的模型输出目录，确认显卡和迭代数，开始训练。训练完成后，模型保存到输出目录下的 `point_cloud/iteration_<N>/point_cloud.ply`。
4. 完成后点「自由 3D」，拖动环绕、滚轮缩放，右键或按住 Shift 拖动以平移。用「相机视角」可查看按拍摄位姿生成的渲染并对照原图。点云视图可切换主体和全景。

已有去畸变数据时，选择「已准备的 3DGS 数据集」，填入包含 `images/` 和 `sparse/0/` 的目录即可。

## 运行环境

- 服务默认监听 `127.0.0.1`，一次运行一个任务。使用 `--lan` 时，指定的网卡和子网也可访问。
- `serve.py` 会识别项目中已构建的 COLMAP、`web/.colmap-env/bin/colmap`、`PATH` 中的 `colmap` 和 `web/.colmap-apt/colmap`。本机后备环境使用 COLMAP 3.9.1 做 CPU 增量建图，3DGS 训练使用 RTX 5090。用 `--colmap` 可以指定其他可执行文件。
- 3DGS 默认使用 `web/.venv/bin/python`。用 `--gs-python` 可以指定其他 Python 环境，用 `--gs-root` 可以指定包含 `train.py` 的 3DGS 源码目录。
- 重启服务时可附加 `--restore-training <模型输出目录>`，打开已保存的训练结果和渲染预览。
- RTX 5090 训练环境需要支持 `sm_120` 的 CUDA 版 PyTorch，以及面向该架构编译的 `simple-knn` 和 `diff-gaussian-rasterization`。运行器按选中显卡的 UUID 分配 CUDA 设备，并在训练前执行 CUDA 检查。
- 打开 `index.html` 可以直接操作示例点云并查看 COLMAP 命令。启动本地服务后可以运行任务和读取实际模型。
