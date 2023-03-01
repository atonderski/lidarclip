# assumes you have run
#!pip install -qqq diffusers==0.11.1 transformers ftfy gradio

# imports
import argparse
import os

import torch
from clip.clip import _transform
from PIL import Image
from torchvision.transforms import Compose
from tqdm import tqdm

from lidarclip.helpers import try_paths
from lidarclip.loader import build_loader

VERTICAL_FOV = 40
VERTICAL_ANGLES = (
    15.0,
    11.0,
    8.0,
    5.0,
    3.0,
    2.0,
    1.67,
    1.33,
    1.0,
    0.67,
    0.33,
    0.0,
    -0.33,
    -0.67,
    -1.0,
    -1.33,
    -1.67,
    -2,
    -2.33,
    -2.67,
    -3.0,
    -3.33,
    -3.67,
    -4.0,
    -4.33,
    -4.67,
    -5.0,
    -5.33,
    -5.67,
    -6.0,
    -7.0,
    -8.0,
    -9.0,
    -10.0,
    -11.0,
    -12.0,
    -13.0,
    -14.0,
    -19.0,
    -25.0,
)

HORIZONTAL_RESOLUTION = 0.33


def find_filename(output_dir, idx):
    # name files as args.output_dir/idx_i.png where i is the i-th image generated for that idx
    # try to find the first i that doesn't exist
    i = 0
    while True:
        filename = os.path.join(output_dir, f"{idx}_{i}.png")
        if not os.path.exists(filename):
            return filename
        i += 1


def save_images(output_dir, images, idx):
    # images is a list of PIL images

    # mkdir
    os.makedirs(output_dir, exist_ok=True)

    filenames = [find_filename(output_dir, i) for i in idx]

    for i, img in enumerate(images):
        img.save(filenames[i])


def main(args):
    # mkdir
    os.makedirs(args.output_dir, exist_ok=True)

    if len(args.range_view_output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    transform = _transform(args.resolution)
    # skip normalization and "to tensor"
    transform = Compose(transform.transforms[:-2])
    dataset_path = try_paths(
        "/home/s0001396/Documents/phd/datasets/once", "/mimer/NOBACKUP/groups/clippin/datasets/once"
    )
    # init dataset
    loader = build_loader(
        once_datadir=dataset_path,
        clip_preprocess=transform,
        once_split=args.split,
        dataset_name="once",
    )
    dataset = loader.dataset
    # Get indexes
    with open(args.indexes_path, "r") as f:
        indexes = [int(x) for x in f.readlines()]

    # Check if image already exists
    if not args.regenerate:  # if we don't want to regenerate, remove indexes that already exist
        existing_files = os.listdir(args.output_dir)
        indexes = [idx for idx in indexes if f"{idx}_0.png" not in existing_files]

    # batch index and loop with tqdm
    for i in tqdm(range(0, len(indexes)), desc="Generating images"):
        idx = indexes[i]
        image, pc = dataset.get_index_without_lidar_proj(idx)

        # Save images
        save_images(args.output_dir, [image], [idx])

        if len(args.range_view_output_dir):
            pc[:, 0] = pc[:, 0] - (1920 / 2 - 1080 / 2)
            pc[:, 0] = (pc[:, 0] / (1080 / args.resolution)).ceil()
            pc[:, 1] = (pc[:, 1] / (1080 / args.resolution)).ceil()
            pc[:, 2] = pc[:, 2] / 150
            pc[:, 3] = pc[:, 3] / 100

            coordinates = pc[:, 0].view([1, -1]) * args.resolution + pc[:, 1].view([1, -1])
            coord_max = args.resolution**2 - 1

            masked_points = (
                (pc[:, 0] >= 0)
                * (pc[:, 0] <= args.resolution - 1)
                * (pc[:, 1] >= 0)
                * (pc[:, 1] <= args.resolution - 1)
                * (pc[:, 3] <= 1.0)
            )

            true_coordinates = coordinates[:]
            true_coordinates[0, ~masked_points] = coord_max
            depth_scatters = torch.zeros(
                [1, args.resolution**2], device=coordinates.device
            ).scatter_add(1, true_coordinates.long(), pc[:, 2:3].view(1, -1))
            depth_scatters = depth_scatters.view(1, args.resolution, args.resolution).T
            intensity_scatters = torch.zeros(
                [1, args.resolution**2], device=coordinates.device
            ).scatter_add(1, true_coordinates.long(), pc[:, 3:].view(1, -1))
            intensity_scatters = intensity_scatters.view(1, args.resolution, args.resolution).T

            range_image = torch.cat([depth_scatters, depth_scatters, intensity_scatters], dim=-1)
            range_image = (range_image * 255).to(torch.uint8)
            range_image = Image.fromarray(range_image.detach().cpu().numpy())

            save_images(args.range_view_output_dir, [range_image], [idx])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/s0001396/Documents/phd/wasp/nlp/lidar-clippin/gen_images/once_val_og",
        choices=[
            "/mimer/NOBACKUP/groups/clippin/gen_images/once_val_og",
            "/home/s0001396/Documents/phd/wasp/nlp/lidar-clippin/gen_images/once_val_og",
        ],
    )
    parser.add_argument(
        "--range-view-output-dir",
        type=str,
        default="/home/s0001396/Documents/phd/wasp/nlp/lidar-clippin/gen_images/once_val_og_lidar",
        choices=[
            "/mimer/NOBACKUP/groups/clippin/gen_images/once_val_og_lidar",
            "/home/s0001396/Documents/phd/wasp/nlp/lidar-clippin/gen_images/once_val_og_lidar",
        ],
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
