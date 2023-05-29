# assumes you have run
#!pip install -qqq diffusers==0.11.1 transformers ftfy gradio

# imports
import argparse
import os

from tqdm import tqdm

import torch

from clip.clip import _transform

from lidarclip.anno_loader import build_anno_loader


def visulize_object(annos, obj_idx):
    import matplotlib.pyplot as plt
    import numpy as np

    points_in_box = annos["points_per_obj"][obj_idx]
    length, w, h = annos["boxes_3d"][obj_idx][3:6].numpy().tolist()

    # Top-view, facing forward
    plt.figure()
    plt.title("Top-view, facing forward")
    plt.scatter(
        -points_in_box[:, 1],
        points_in_box[:, 0],
        s=0.3,
        c=np.clip(points_in_box[:, 3], 0, 1),
        cmap="coolwarm",
    )
    plt.axis([-w / 2, w / 2, -length / 2, length / 2])
    plt.axis("equal")

    # Side-view
    plt.figure()
    plt.title("Left-view")
    plt.scatter(
        -points_in_box[:, 0],
        points_in_box[:, 2],
        s=0.3,
        c=np.clip(points_in_box[:, 3], 0, 1),
        cmap="coolwarm",
    )
    plt.axis([-length / 2, length / 2, -h / 2, h / 2])
    plt.axis("equal")

    # Side-view
    plt.figure()
    plt.title("Right-view")
    plt.scatter(
        points_in_box[:, 0],
        points_in_box[:, 2],
        s=0.3,
        c=np.clip(points_in_box[:, 3], 0, 1),
        cmap="coolwarm",
    )
    plt.axis([-length / 2, length / 2, -h / 2, h / 2])
    plt.axis("equal")

    # Front-view
    plt.figure()
    plt.title("Front-view")
    plt.scatter(
        points_in_box[:, 1],
        points_in_box[:, 2],
        s=0.3,
        c=np.clip(points_in_box[:, 3], 0, 1),
        cmap="coolwarm",
    )
    plt.axis([-w / 2, w / 2, -h / 2, h / 2])
    plt.axis("equal")

    plt.show()


def main(args):
    # mkdir
    os.makedirs(args.output_dir, exist_ok=True)
    transform = _transform(256)
    # init dataset
    loader = build_anno_loader(
        datadir=args.data_dir,
        clip_preprocess=transform,
        split=args.split,
        num_workers=1,
        return_points_per_obj=True,
    )

    out = []
    i = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            i = i + 1
            images, point_clouds, meta_info, annos = batch

            if args.visulize_objects:
                for anno in annos:
                    for obj_idx in range(len(anno["boxes_3d"])):
                        visulize_object(anno, obj_idx)

            out.extend(annos)

            if i % 100 == 0:
                torch.save(out, os.path.join(args.output_dir, f"{args.split}_{i}.pt"))

    torch.save(out, os.path.join(args.output_dir, f"{args.split}.pt"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/s0001396/Documents/phd/datasets/once/cls",
        help="Output directory",
        choices=[
            "/mimer/NOBACKUP/groups/clippin/datasets/once/cls",
            "/home/s0001396/Documents/phd/datasets/once/cls",
        ],
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/s0001396/Documents/phd/datasets/once",
        help="Data directory",
        choices=[
            "/mimer/NOBACKUP/groups/clippin/datasets/once",
            "/home/s0001396/Documents/phd/datasets/once",
        ],
    )
    parser.add_argument("--split", type=str, default="val", help="Split", choices=["val", "train"])
    parser.add_argument("--visulize-objects", action="store_true", help="Visulize objects")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
