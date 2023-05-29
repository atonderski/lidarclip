import argparse
import os.path as osp
import math
from collections import defaultdict

import clip
import torch
from einops import rearrange
from skimage.draw import polygon2mask
from tqdm import tqdm
import numpy as np

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


def vit_attention_no_pooling(model, x):
    seq_len, n, d = x.shape
    attn_mask = (1 - torch.eye(seq_len, device=x.device)).bool()
    return model(x, x, x, need_weights=False, attn_mask=attn_mask)[0]


def resnet_attention_no_pooling(model, x):
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
        2, 0, 1
    )  # NCHW -> (HW)NC
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
    x = x + model.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
    x = x[1:]  # remove the "fake" averaged token (HW+1)NC -> (HW)NC
    return model.c_proj(model.v_proj(x))  # (HW)NC


def clip_vit_forward_no_pooling(model, x):
    x = model.conv1(x.half())  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat(
        [
            model.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ],
        dim=1,
    )  # shape = [*, grid ** 2 + 1, width]
    x = x + model.positional_embedding.to(x.dtype)
    x = model.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND

    #### START REPLACE TRANSFORMER #####
    # Only do first n-1 layers
    for blk in model.transformer.resblocks[:-1]:
        x = blk(x)
    last_res_block = model.transformer.resblocks[-1]
    x = x + vit_attention_no_pooling(last_res_block.attn, last_res_block.ln_1(x))
    x = x + last_res_block.mlp(last_res_block.ln_2(x))
    #### END REPLACE TRANSFORMER #####

    x = x.permute(1, 0, 2)  # LND -> NLD

    x = model.ln_post(x[:, 1:, :])

    if model.proj is not None:
        x = x @ model.proj

    return x


def clip_resnet_forward_no_pooling(model, x):
    def stem(x):
        x = model.relu1(model.bn1(model.conv1(x)))
        x = model.relu2(model.bn2(model.conv2(x)))
        x = model.relu3(model.bn3(model.conv3(x)))
        x = model.avgpool(x)
        return x

    x = x.type(model.conv1.weight.dtype)
    x = stem(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = resnet_attention_no_pooling(model.attnpool, x)
    x = x.permute(1, 0, 2)  # (HW)ND -> N(HW)D

    return x


def load_model(args):
    clip_model, clip_preprocess = clip.load(args.clip_version)
    clip_model.to(DEVICE)
    return clip_model, clip_preprocess


def main(args):
    # assert torch.cuda.is_available()

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

    obj_feats_path = f"{args.prefix}_2d_objs.pt"
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
            images = images.to(DEVICE)
            if "vit" in args.clip_version.lower():
                image_features = clip_vit_forward_no_pooling(model.visual, images)
            else:
                image_features = clip_resnet_forward_no_pooling(model.visual, images)
            n, seq_len, d = image_features.shape
            h = w = int(math.sqrt(seq_len))
            image_features = rearrange(image_features, "n (h w) c -> n h w c", h=h, w=w)
            for i, image_feature in enumerate(image_features):
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
                    box2d = [x // (images.shape[-1] / h) for x in box2d]
                    obj_feat = _extract_obj_feat(box2d, image_feature)
                    obj_feats[name].append(obj_feat.clone())
            if batch_i == 0:
                debug_path = f"{args.prefix}_2d_objs_bev_debug.pt"
                if not osp.exists(debug_path):
                    torch.save(image_features[0:1].clone(), debug_path)
    torch.save(obj_feats, obj_feats_path)


def _extract_obj_feat(box2d, bev_feature):
    x_dim, y_dim, _ = bev_feature.shape
    # Convert box2d to a polygon
    box_points = torch.tensor(
        [
            [box2d[0], box2d[1]],
            [box2d[2], box2d[1]],
            [box2d[2], box2d[3]],
            [box2d[0], box2d[3]],
        ]
    )
    # Create a mask of the pixels that are within the bounding box
    # Flip x and y (since y is height and x is width)
    mask = polygon2mask((x_dim, y_dim), box_points.cpu().numpy())

    # Pool the features within the bounding box
    pooled_features = bev_feature[torch.tensor(mask, device=bev_feature.device)].mean(dim=(0))
    return pooled_features


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
