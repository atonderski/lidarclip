import argparse
import os.path as osp
from collections import defaultdict

import clip
import torch
from einops import rearrange
from mmcv.runner import load_checkpoint
from skimage.draw import polygon2mask
from tqdm import tqdm

from lidarclip.anno_loader import build_anno_loader
from lidarclip.model.sst import LidarEncoderSST
from train import LidarClip

VOXEL_SIZE = 0.5
DISTANCE = 40  # Discard objects beyond this distance

DEFAULT_DATA_PATHS = {
    "once": "/once",
    "nuscenes": "/proj/berzelius-2021-92/data/nuscenes",
}


def load_model(args):
    clip_model, clip_preprocess = clip.load(args.clip_version)
    lidar_encoder = LidarEncoderSST(
        "lidarclip/model/sst_encoder_only_config.py", clip_model.visual.output_dim
    )
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
    )

    obj_feats_path = f"{args.prefix}_lidar_objs.pt"
    if osp.exists(obj_feats_path):
        print("Found existing file, skipping")
        return

    print("Starting feature generation...")
    obj_feats = defaultdict(list)
    with torch.no_grad():
        for batch_i, batch in tqdm(
            enumerate(loader), desc="Generating features", total=len(loader)
        ):
            _, point_clouds, _, annos = batch
            point_clouds = [pc.to("cuda") for pc in point_clouds]
            lidar_features, _ = model.lidar_encoder(point_clouds, no_pooling=True)
            bev_features = rearrange(lidar_features, "(h w) n c -> n h w c", h=80, w=80)
            for i, bev_feature in enumerate(bev_features):
                for name, box3d in zip(annos[i]["names"], annos[i]["boxes_3d"]):
                    if box3d[:2].norm() > DISTANCE:
                        continue
                    obj_feat = _extract_obj_feat(box3d, bev_feature)
                    obj_feats[name].append(obj_feat.clone())
            if batch_i == 0:
                debug_path = f"{args.prefix}_lidar_objs_bev_debug.pt"
                if not osp.exists(debug_path):
                    torch.save(bev_features[0:1].clone(), debug_path)
    torch.save(obj_feats, obj_feats_path)


def _extract_obj_feat(box3d, bev_feature):
    y_dim, x_dim, _ = bev_feature.shape
    # Convert box to feature grid space
    box_center = box3d[:2] / VOXEL_SIZE
    # Compensate y coordinate for the fact that the lidar features are
    # centered around the ego vehicle in the y direction (x starts from 0)
    box_center[1] += y_dim / 2
    box_size = box3d[3:5] / VOXEL_SIZE
    box_rotation = -torch.Tensor([box3d[6]])

    # Create the corner points of the bounding box
    box_points = torch.tensor(
        [
            [-box_size[0] / 2, -box_size[1] / 2],
            [-box_size[0] / 2, box_size[1] / 2],
            [box_size[0] / 2, box_size[1] / 2],
            [box_size[0] / 2, -box_size[1] / 2],
        ]
    )
    # Create a rotation matrix from the box rotation
    rotation_matrix = torch.tensor(
        [
            [torch.cos(box_rotation), -torch.sin(box_rotation)],
            [torch.sin(box_rotation), torch.cos(box_rotation)],
        ]
    )
    # Rotate the corner points of the bounding box
    box_points = torch.matmul(box_points, rotation_matrix) + box_center
    # Create a mask of the pixels that are within the bounding box
    # Flip x and y (since y is height and x is width)
    mask = polygon2mask((x_dim, y_dim), box_points.cpu().numpy()[:, ::-1])
    # Pool the features within the bounding box
    pooled_features = bev_feature[mask].mean(dim=(0))
    return pooled_features


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
