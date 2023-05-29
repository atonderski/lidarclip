import argparse
import os.path as osp
from collections import defaultdict

import clip
import torch
from tqdm import tqdm

from lidarclip.anno_loader import build_anno_loader

IMAGE_X_MIN = 420
IMAGE_Y_MIN = 0
IMAGE_X_MAX = 420 + 1080
IMAGE_Y_MAX = 1080
IMAGE_SCALING = 1080 // 224
VOXEL_SIZE = 1
MIN_DISTANCE = 0
MAX_DISTANCE = 40  # Discard objects beyond this distance

DEFAULT_DATA_PATHS = {
    "once": "/once",
    "nuscenes": "/proj/berzelius-2021-92/data/nuscenes",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(args):
    clip_model, clip_preprocess = clip.load(args.clip_version)
    clip_model.to(DEVICE)
    return clip_model, clip_preprocess


def main(args):
    # assert torch.cuda.is_available()

    print("Loading model...")
    model, clip_preprocess = load_model(args)

    def dummy_transform(x):
        return x

    print("Setting up dataloader...")
    loader = build_anno_loader(
        args.data_path,
        dummy_transform,
        batch_size=args.batch_size,
        num_workers=8,
        split=args.split,
        dataset_name=args.dataset_name,
    )

    obj_feats_path = f"{args.prefix}_2d_objs_highres.pt"
    if osp.exists(obj_feats_path):
        print("Found existing file, skipping")
        return

    print("Starting feature generation...")
    obj_feats = defaultdict(list)
    with torch.no_grad():
        for batch_i, batch in tqdm(
            enumerate(loader), desc="Generating features", total=len(loader)
        ):
            images, annos = batch[0], batch[3]
            for i, image in enumerate(images):
                for name, box2d, box3d in zip(
                    annos[i]["names"], annos[i]["boxes_2d"], annos[i]["boxes_3d"]
                ):
                    if box3d[:2].norm() > MAX_DISTANCE or box3d[:2].norm() < MIN_DISTANCE:
                        continue
                    # crop image to bounding box
                    cropped_image = image.crop(box2d)
                    # apply transform
                    cropped_image = clip_preprocess(cropped_image).unsqueeze(0)
                    obj_feat = model.encode_image(cropped_image.to(DEVICE)).squeeze()
                    obj_feats[name].append(obj_feat.clone())

    torch.save(obj_feats, obj_feats_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
