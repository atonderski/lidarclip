# assumes you have run
#!pip install -qqq diffusers==0.11.1 transformers ftfy gradio

# imports
import argparse
import os

from PIL import Image
from tqdm import tqdm

import torch
from torchvision.transforms import Compose

from clip.clip import _transform

from lidarclip.loader import build_loader


def find_filename(args, idx):
    # name files as args.output_dir/idx_i.png where i is the i-th image generated for that idx
    # try to find the first i that doesn't exist
    i = 0
    while True:
        filename = os.path.join(args.output_dir, f"{idx}_{i}.png")
        if not os.path.exists(filename):
            return filename
        i += 1


def save_images(args, images, idx):
    # images is a list of PIL images

    # mkdir
    os.makedirs(args.output_dir, exist_ok=True)

    filenames = [find_filename(args, i) for i in idx]

    for i, img in enumerate(images):
        img.save(filenames[i])


def main(args):
    # mkdir
    os.makedirs(args.output_dir, exist_ok=True)
    transform = _transform(args.resolution)
    # skip normalization and "to tensor"
    transform = Compose(transform.transforms[:-2])
    # init dataset
    loader = build_loader(
        datadir="/mimer/NOBACKUP/groups/clippin/datasets/once",
        clip_preprocess=transform,
        split=args.split,
    )
    dataset = loader.dataset
    # Get indexes
    with open(args.indexes_path, "r") as f:
        indexes = [int(x) for x in f.readlines()]

    # Check if image already exists
    if not args.regenerate:  # if we don't want to regenerate, remove indexes that already exist
        existing_files = os.listdir(args.output_dir)
        indexes = [idx for idx in indexes if not f"{idx}_0.png" in existing_files]

    # batch index and loop with tqdm
    for i in tqdm(range(0, len(indexes)), desc="Generating images"):
        idx = indexes[i]
        image, _ = dataset[idx]
        # Save images
        save_images(args, [image], [idx])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="/mimer/NOBACKUP/groups/clippin/gen_images/once_val_og"
    )
    parser.add_argument(
        "--indexes_path",
        type=str,
        default="/mimer/NOBACKUP/groups/clippin/gen_images/once_val_ids.txt",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--regenerate", type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
