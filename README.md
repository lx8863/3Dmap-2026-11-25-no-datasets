# 3Dmap

COLMAP 与 3D Gaussian Splatting 本地工作台，归档日期：2026-11-25。

本仓库导入自 `3Dmap-2026-11-25-no-datasets.zip`。原始数据集和点云文件不包含在仓库中；其余源码、工作台、训练配置与日志保留。点云由使用者在本地重建或训练生成。

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
- `outputs/`：归档包中的训练配置、相机参数与日志，已排除点云。

## 启动工作台

按 `web/README.md` 准备依赖和运行环境后，在仓库根目录运行：

```bash
python3 -B web/serve.py
```

然后访问 <http://127.0.0.1:8765/>。归档中的环境路径、系统依赖和显卡配置可能需要按运行机器调整。

## 许可

各子项目沿用其原有许可证；本次归档不改变 COLMAP、Gaussian Splatting 或其他第三方代码的许可条件，具体见对应目录内的许可证文件。
