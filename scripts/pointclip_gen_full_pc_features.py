import argparse
import os

from pointclip_utils import PointCLIP_ZS

import torch

import clip

from lidarclip.anno_loader import CLASSES
from lidarclip.anno_loader import build_anno_loader
from lidarclip.loader import build_loader as build_dataonly_loader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="/mimer/NOBACKUP/groups/clippin/features/pointclip",
        help="Output filename",
        choices=[
            "/mimer/NOBACKUP/groups/clippin/features",
        ],
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/mimer/NOBACKUP/groups/clippin/datasets/once/",
        help="Data directory",
        choices=[
            "/mimer/NOBACKUP/groups/clippin/datasets/once/",
            "/home/s0001396/Documents/phd/datasets/once/",
        ],
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-L/14",
        help="CLIP model",
        choices=["ViT-B/32", "RN50", "RN101", "RN50x4", "RN50x16", "ViT-L/14", "ViT-B/16"],
    )
    parser.add_argument("--num-views", type=int, default=6, help="Number of views")
    parser.add_argument("--split", type=str, default="val", help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--dataset-name", type=str, default="once", help="Dataset name")
    parser.add_argument("--use_anno_loader", action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    # create output directory
    os.makedirs(os.path.basename(args.output), exist_ok=True)
    # load clip model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load(args.clip_model, device=device)

    # build pointclip model
    pointclip_zs = PointCLIP_ZS()
    pointclip_zs.build_model(clip_model, CLASSES, args.num_views)

    build_loader = build_anno_loader if args.use_anno_loader else build_dataonly_loader
    loader = build_loader(
        args.data_path,
        clip_preprocess,
        batch_size=args.batch_size,
        num_workers=8,
        split=args.split,
        dataset_name=args.dataset_name,
    )
    with torch.no_grad():
        for batch in tqdm(loader):
            images, point_clouds = batch[:2]
            for pc in point_clouds:
                pc = pc.to(device)

                normalizing_factor = pc.max() / 2
                # normalize pc to [-1,1]
                pc[:, :3] = pc[:, :3] / normalizing_factor

                # pc is forward, left, up
                # modelnet has left, up, backwards
                # so we need to swap axes
                pc = pc[:, [1, 2, 0]]
                # and flip x axis
                pc[:, 2] = -pc[:, 2]

                pc.unsqueeze_(0)

                _ = pointclip_zs.model_inference(pc[..., :3])

    print("Finished processing all samples")

    filename = f"features_fullPC_{args.clip_model}.pt"
    torch.save(torch.cat(pointclip_zs.feat_store, dim=0), os.path.join(args.output_dir, filename))


if __name__ == "__main__":
    args = parse_args()
    main(args)
