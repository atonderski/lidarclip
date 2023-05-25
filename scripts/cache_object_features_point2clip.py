import argparse
import os.path as osp
from collections import defaultdict

import clip
import math
import torch
import numpy as np
from einops import rearrange
from mmcv.runner import load_checkpoint
from tqdm import tqdm

from lidarclip.anno_loader import build_anno_loader
from cache_object_features_2d import clip_vit_forward_no_pooling, _extract_obj_feat
from train import LidarClip

try:
    from lidarclip.model.depth import DepthEncoder
except ImportError:
    print("Got ImportError for Depth model and loss, not importing")

IMAGE_X_MIN = 420
IMAGE_Y_MIN = 0
IMAGE_X_MAX = 420 + 1080
IMAGE_Y_MAX = 1080
IMAGE_SCALING = 1080 // 224
MIN_DISTANCE = 0
MAX_DISTANCE = 40  # Discard objects beyond this distance

DEFAULT_DATA_PATHS = {
    "once": "/once",
    "nuscenes": "/proj/berzelius-2021-92/data/nuscenes",
}


def load_model(args):
    clip_model, clip_preprocess = clip.load(args.clip_version)
    lidar_encoder = DepthEncoder(args.clip_version, True)
    model = LidarClip(lidar_encoder, clip_model, 1, 1)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to("cuda")
    return model, clip_preprocess


def main(args):
    assert torch.cuda.is_available()
    print("Loading model...")
    model, clip_preprocess = load_model(args)
    print("Setting up dataloader...")
    loader = build_anno_loader(
        args.data_path,
        clip_preprocess,
        batch_size=args.batch_size,
        num_workers=8,
        split=args.split,
        dataset_name=args.dataset_name,
        depth_rendering="aug",
    )

    obj_feats_path = f"{args.prefix}_clip2point_objs.pt"
    if osp.exists(obj_feats_path):
        print("Found existing file, skipping")
        return

    print("Starting feature generation...")
    obj_feats = defaultdict(list)
    with torch.no_grad():
        for batch_i, batch in tqdm(
            enumerate(loader), desc="Generating features", total=len(loader)
        ):
            point_clouds, annos = batch[1:3]
            point_clouds = [pc.to("cuda") for pc in point_clouds]
            cam_intrinsics = torch.stack([anno["cam_intrinsic"].to("cuda") for anno in annos])
            img_size = torch.stack([anno["img_size"].to("cuda") for anno in annos])

            rendered_point_clouds = model.lidar_encoder.render_depth(
                point_clouds, cam_intrinsics, img_size, False
            )

            rendered_point_clouds = rearrange(rendered_point_clouds, "b v c h w -> (b v) c h w")

            point_cloud_features = clip_vit_forward_no_pooling(
                model.lidar_encoder._clip.visual, rendered_point_clouds
            )

            n, seq_len, d = point_cloud_features.shape
            h = w = int(math.sqrt(seq_len))
            point_cloud_features = rearrange(point_cloud_features, "n (h w) c -> n h w c", h=h, w=w)
            for i, point_cloud_feature in enumerate(point_cloud_features):
                for name, box2d, box3d in zip(
                    annos[i]["names"], annos[i]["boxes_2d"], annos[i]["boxes_3d"]
                ):
                    if box3d[:2].norm() > MAX_DISTANCE or box3d[:2].norm() < MIN_DISTANCE:
                        continue
                    # clamp to image bounds and remove offset
                    box2d[0] = np.clip(box2d[0], IMAGE_X_MIN, IMAGE_X_MAX - 1) - IMAGE_X_MIN
                    box2d[1] = np.clip(box2d[1], IMAGE_Y_MIN, IMAGE_Y_MAX - 1)
                    box2d[2] = np.clip(box2d[2], IMAGE_X_MIN, IMAGE_X_MAX - 1) - IMAGE_X_MIN
                    box2d[3] = np.clip(box2d[3], IMAGE_Y_MIN, IMAGE_Y_MAX - 1)
                    # scale with transform downsampling
                    box2d = [x / (IMAGE_SCALING) for x in box2d]
                    # scale with feature resolution
                    box2d = [x // (rendered_point_clouds.shape[-1] / h) for x in box2d]
                    obj_feat = _extract_obj_feat(box2d, point_cloud_feature)
                    obj_feats[name].append(obj_feat.clone())
            if batch_i == 0:
                debug_path = f"{args.prefix}_clip2point_objs_debug.pt"
                if not osp.exists(debug_path):
                    torch.save(point_cloud_features[0:1].clone(), debug_path)
    torch.save(obj_feats, obj_feats_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Full path to the checkpoint file"
    )
    parser.add_argument("--clip-version", type=str, default="ViT-L/14")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--prefix", type=str, default="/features/cached")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset-name", type=str, default="once", choices=["once", "nuscenes"])
    args = parser.parse_args()
    if not args.data_path:
        args.data_path = DEFAULT_DATA_PATHS[args.dataset_name]
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # VOXEL_SIZE = 1
    # test_feat = torch.ones((100, 100, 3))
    # # Draw a diagonal narrow box with value 2 from 24,78 to 36,90
    # for i in range(-5, 5):
    #     test_feat[30 + i, 80 + i, :] = 2
    # test_box = torch.Tensor([30, 30, 123012, 10, 1, 1239123, 3 * torch.pi / 4])
    # feat = _extract_obj_feat(test_box, test_feat)
    # print(feat)
    # import matplotlib.pyplot as plt

    # plt.imshow(test_feat[..., 0])
    # plt.show()
    # plt.show()
