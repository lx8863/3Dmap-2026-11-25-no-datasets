# 3Dmap

3Dmap 是本地照片重建工作台。通过 VGGT 或 COLMAP 标定相机，再用 3D Gaussian Splatting 训练、预览并导出 PLY。

仓库从 2026-11-25 的归档发展而来。首页内置一个「庭院花簇」示例，模型和来源说明位于 `web/sample/`。导入照片生成的项目会保存在本机。

## 下载

在 GitHub 仓库页面点击 **Code → Download ZIP**，下载后解压；也可以使用 Git：

```bash
git clone https://github.com/lx8863/3Dmap-2026-11-25-no-datasets.git
cd 3Dmap-2026-11-25-no-datasets
```

不需要 Git LFS。

## 目录

- `web/`：网页工作台和本地 Python 服务，使用说明见 [web/README.md](web/README.md)。
- `colmap/`：COLMAP 源码。
- `gaussian-splatting/gaussian-splatting/`：3D Gaussian Splatting 源码及依赖代码。
- `outputs/`：原始归档的训练配置、相机参数和日志。
- `packaging/`：Windows 便携版启动器和构建脚本。

## 启动工作台

Windows 双击 `Start-3Dmap.cmd`。也可以在仓库根目录运行 `python -B web/serve.py --open`，进入照片多选、相机标定、3DGS 训练、渲染和 PLY 下载的工作台。环境配置见 [工作台说明](web/README.md)。

## Windows 便携版

在已按[工作台说明](web/README.md)配置 VGGT 和 3DGS 的 Windows 环境中，用 VGGT 虚拟环境运行打包脚本。

```powershell
web/.vggt/venv/Scripts/python.exe -B packaging/build_windows.py
```

发布目录为 `dist/3Dmap-Windows-x64/`。将整个目录复制到目标电脑，然后双击 `3Dmap.exe`。预编译 CUDA 扩展面向 RTX 50 系列。运行所需的驱动及操作说明随包提供。

## 许可

各子项目沿用其原有许可证；本次归档不改变 COLMAP、Gaussian Splatting 或其他第三方代码的许可条件，具体见对应目录内的许可证文件。
