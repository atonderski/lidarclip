import argparse
import os

from mmcv.runner import load_checkpoint
from tqdm import tqdm

import torch

import clip

from lidarclip.anno_loader import build_anno_loader
from lidarclip.loader import build_loader as build_dataonly_loader
from lidarclip.model.sst import LidarEncoderSST

from train import LidarClip


DEFAULT_DATA_PATHS = {
    "once": "/proj/nlp4adas/datasets/once",
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
    model, clip_preprocess = load_model(args)
    build_loader = build_anno_loader if args.use_anno_loader else build_dataonly_loader
    loader = build_loader(
        args.data_path,
        clip_preprocess,
        batch_size=args.batch_size,
        num_workers=8,
        split=args.split,
        dataset_name=args.dataset_name,
    )

    img_path, lidar_path = (
        f"{args.prefix}_img.pt",
        f"{args.prefix}_lidar.pt",
    )
    if os.path.exists(img_path) or os.path.exists(lidar_path):
        print("Found existing files, skipping")
        return

    img_feats = []
    lidar_feats = []
    with torch.no_grad():
        for batch in tqdm(loader):
            images, point_clouds = batch[:2]
            point_clouds = [pc.to("cuda") for pc in point_clouds]
            images = [img.to("cuda") for img in images]
            images = torch.cat([i.unsqueeze(0) for i in images])
            image_features = model.clip.encode_image(images)
            lidar_features, _ = model.lidar_encoder(point_clouds)
            img_feats.append(image_features.detach().cpu())
            lidar_feats.append(lidar_features.detach().cpu())

    img_feats = torch.cat(img_feats, dim=0)
    lidar_feats = torch.cat(lidar_feats, dim=0)

    torch.save(img_feats, img_path)
    torch.save(lidar_feats, lidar_path)


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
    parser.add_argument("--use-anno-loader", action="store_true")
    parser.add_argument("--dataset-name", type=str, default="once", choices=["once", "nuscenes"])
    args = parser.parse_args()
    if not args.data_path:
        args.data_path = DEFAULT_DATA_PATHS[args.dataset_name]
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
