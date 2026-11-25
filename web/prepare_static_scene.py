"""Prepare exclusion masks and static sparse seeds for a synchronized COLMAP scene."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2


TRAFFIC_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
HOLDOUT_STRIDE = 8


@torch.inference_mode()
def prepare_masks(dataset, names):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=False, box_score_thresh=0.38).eval().cuda()
    preprocess = weights.transforms()
    class_ids = torch.tensor([
        index for index, name in enumerate(weights.meta["categories"]) if name in TRAFFIC_CLASSES
    ], device="cuda")
    mask_root = dataset / "static_masks"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for index, name in enumerate(names, 1):
        target = (mask_root / name).with_suffix(".png")
        if not target.exists():
            with Image.open(dataset / "images" / name) as source:
                image = source.convert("RGB")
            sample = preprocess(image).cuda()
            prediction = model([sample])[0]
            traffic = torch.isin(prediction["labels"], class_ids)
            excluded = (prediction["masks"][traffic, 0] >= 0.4).any(dim=0).byte().cpu().numpy()
            valid = (1 - cv2.dilate(excluded, kernel)) * 255
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp.png")
            if not cv2.imwrite(str(temporary), valid):
                raise OSError(f"Could not write {target}")
            temporary.replace(target)
        if index % 50 == 0 or index == len(names):
            print(f"Prepared {index}/{len(names)} traffic masks", flush=True)


def prepare_sparse_seeds(dataset, images, holdout):
    from scene.dataset_readers import storePly
    from utils.read_write_model import read_points3D_binary

    model = dataset / "sparse" / "0"
    supported = {}
    excluded = set()
    for image in images.values():
        if image.name in holdout:
            continue
        mask_file = (dataset / "static_masks" / image.name).with_suffix(".png")
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(f"Could not read {mask_file}")
        height, width = mask.shape
        x, y = np.floor(image.xys).astype(np.int64).T
        valid = (image.point3D_ids >= 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        point_ids = image.point3D_ids[valid]
        background = mask[y[valid], x[valid]] != 0
        excluded.update(point_ids[~background].tolist())
        for point_id in point_ids[background].tolist():
            supported[point_id] = supported.get(point_id, 0) + 1

    points = read_points3D_binary(model / "points3D.bin")
    seeds = [point for point_id, point in points.items()
             if supported.get(point_id, 0) >= 2 and point_id not in excluded]
    if not seeds:
        raise ValueError("No static sparse points remain after masking")
    storePly(str(model / "points3D.ply"), np.stack([point.xyz for point in seeds]),
             np.stack([point.rgb for point in seeds]))
    print(f"Retained {len(seeds)}/{len(points)} sparse seeds from training views", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.gs_root.resolve()))
    from scene.colmap_loader import read_extrinsics_binary

    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    dataset = args.dataset.resolve()
    sparse = dataset / "sparse" / "0"
    images = read_extrinsics_binary(sparse / "images.bin")
    names_by_camera = {}
    for image in images.values():
        camera, filename = Path(image.name).parts
        names_by_camera.setdefault(camera, {})[Path(filename).stem] = image.name
    timestamps = sorted(set.intersection(*(set(names) for names in names_by_camera.values())))
    if len(names_by_camera) < 2 or not timestamps or any(len(names) != len(timestamps) for names in names_by_camera.values()):
        raise ValueError("Expected multiple cameras with matching timestamps")
    holdout = {names[stamp] for names in names_by_camera.values() for stamp in timestamps[::HOLDOUT_STRIDE]}
    print(f"Preparing {len(images) - len(holdout)} training and {len(holdout)} held-out images", flush=True)
    prepare_masks(dataset, sorted(image.name for image in images.values()))
    prepare_sparse_seeds(dataset, images, holdout)
    (sparse / "test.txt").write_text("\n".join(sorted(holdout)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
