import argparse
import os

import clip
import torch
from tqdm import tqdm

from lidarclip.anno_loader import build_anno_loader
from lidarclip.loader import build_loader as build_dataonly_loader
from lidarclip.model.sst import LidarEncoderSST
from lidarclip.model.depth import DepthEncoder
from train import LidarClip

DEFAULT_DATA_PATHS = {
    "once": "/proj/adas-data/data/once",
    "nuscenes": "/proj/adas-data/data/nuscenes",
}

"""
DEFAULT_DATA_PATHS = {
    "once": "/mimer/NOBACKUP/groups/clippin/datasets/once",
    "nuscenes": "/mimer/NOBACKUP/groups/clippin/datasets/nuscenes",
}
"""


def load_model(args):
    clip_model, clip_preprocess = clip.load(args.clip_version)

    if args.model == "depth":
        lidar_encoder = DepthEncoder(args.clip_version, depth_aug=False)
    elif args.model == "sst":
        lidar_encoder = LidarEncoderSST(
            "lidarclip/model/sst_encoder_only_config.py", clip_model.visual.output_dim
        )
    else:
        raise ValueError(f"Unknown model {args.model}")
    model = LidarClip.load_from_checkpoint(
        args.checkpoint,
        strict=False,
        lidar_encoder=lidar_encoder,
        clip_model=clip_model,
        batch_size=1,
        epoch_size=1,
    )
    # load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to("cuda")
    return model, clip_preprocess


def main(args):
    assert torch.cuda.is_available()
    model, clip_preprocess = load_model(args)
    build_loader = build_anno_loader if args.use_anno_loader else build_dataonly_loader
    loader = build_loader(
        clip_preprocess=clip_preprocess,
        batch_size=args.batch_size,
        num_workers=8,
        once_datadir=args.once_datadir,
        once_split=args.split,
        nuscenes_datadir=args.nuscenes_datadir,
        nuscenes_split=args.split,
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
            images, point_clouds, metadata = batch[:3]
            point_clouds = [pc.to("cuda") for pc in point_clouds]
            # images = [img.to("cuda") for img in images]
            # images = torch.cat([i.unsqueeze(0) for i in images])
            if not args.no_save_img:
                images = images.to("cuda")
                image_features = model.clip.encode_image(images)
                img_feats.append(image_features.detach().cpu())
            lidar_features, _ = model.lidar_encoder(
                point_clouds, metadata=metadata, no_pooling=args.no_pooling
            )
            lidar_feats.append(lidar_features.detach().cpu())

    lidar_feats = torch.cat(lidar_feats, dim=0)
    torch.save(lidar_feats, lidar_path)

    if not args.no_save_img:
        img_feats = torch.cat(img_feats, dim=0)
        torch.save(img_feats, img_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Full path to the checkpoint file"
    )
    parser.add_argument("--model", type=str, default="sst", choices=["sst", "depth"])
    parser.add_argument("--clip-version", type=str, default="ViT-L/14")
    parser.add_argument("--once-datadir", type=str, default=None)
    parser.add_argument("--nuscenes-datadir", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--prefix", type=str, default="/features/cached")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-anno-loader", action="store_true")
    parser.add_argument("--dataset-name", type=str, default="once", choices=["once", "nuscenes"])
    parser.add_argument("--no-pooling", action="store_true")
    parser.add_argument("--no-save-img", action="store_true")
    args = parser.parse_args()
    if not args.once_datadir:
        args.once_datadir = DEFAULT_DATA_PATHS["once"]
    if not args.nuscenes_datadir:
        args.nuscenes_datadir = DEFAULT_DATA_PATHS["nuscenes"]
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
