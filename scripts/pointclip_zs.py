import argparse
import os

import numpy as np
from pointclip_utils import PointCLIP_ZS

import torch

import clip

from lidarclip.anno_loader import CLASSES
import math

DISTANCE_THRESH = 400
POINT_THRESH = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mimer/NOBACKUP/groups/clippin/datasets/once/cls",
        help="Output directory",
        choices=[
            "/mimer/NOBACKUP/groups/clippin/datasets/once/cls",
            "/home/s0001396/Documents/phd/datasets/once/cls",
        ],
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mimer/NOBACKUP/groups/clippin/datasets/once/cls/val.pt",
        help="Data directory",
        choices=[
            "/mimer/NOBACKUP/groups/clippin/datasets/once/cls/val.pt",
            "/home/s0001396/Documents/phd/datasets/once/cls/val.pt",
        ],
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        help="CLIP model",
        choices=["ViT-B/32", "RN50", "RN101", "RN50x4", "RN50x16", "ViT-L/14"],
    )
    parser.add_argument("--num-views", type=int, default=6, help="Number of views")
    parser.add_argument(
        "--pre-computed-feat-path", type=str, default="", help="Path to precomputed features to use"
    )
    args = parser.parse_args()
    return args


def print_stats(labels, preds):
    # print accuracy
    print("Accuracy: ", np.sum(np.array(labels) == np.array(preds)) / len(labels))
    # print class-wise accuracy
    print("Class-wise accuracy:")
    for c in range(len(CLASSES)):
        mask = np.array(labels) == c

        print(c, np.sum(np.array(labels)[mask] == np.array(preds)[mask]) / (np.sum(mask) + 1e-6))

    # print confusion matrix using scikit-learn
    print("Confusion matrix:")
    from sklearn.metrics import confusion_matrix

    print(confusion_matrix(labels, preds, labels=list(range(len(CLASSES)))))


def main(args):
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # load data
    data = torch.load(args.data_dir)
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load(args.clip_model, device=device)

    pointclip_zs = PointCLIP_ZS()
    pointclip_zs.build_model(clip_model, CLASSES, args.num_views)
    labels = []
    preds = []

    if len(args.pre_computed_feat_path) > 0:
        print("Loading precomputed features")
        computed_feats = torch.load(args.pre_computed_feat_path)
        if type(computed_feats) == list:
            computed_feats = torch.cat(computed_feats, dim=0)

        pointclip_zs.feat_store = computed_feats

        labels = torch.load(args.pre_computed_feat_path.replace("feat", "label"))
        if type(labels) == list:
            labels = torch.tensor(labels)

        pointclip_zs.label_store = labels

    counter = 0
    mask = []
    for i in range(len(data)):
        anno = data[i]
        for j in range(len(anno["names"])):
            counter += 1

            label = CLASSES.index(anno["names"][j])
            pc = anno["points_per_obj"][j]

            x, y, z, length, w, h, ry = anno["boxes_3d"][j]

            if len(args.pre_computed_feat_path) > 0:
                keep = math.sqrt(x**2 + y**2) < DISTANCE_THRESH
                keep = keep and (len(pc) >= POINT_THRESH)
                mask.append(keep)
                continue

            normalizing_factor = max(length, w, h) / 2
            # normalize pc to [-1,1]
            pc[:, :3] = pc[:, :3] / normalizing_factor

            # pc is forward, left, up
            # modelnet has left, up, backwards
            # so we need to swap axes
            pc = pc[:, [1, 2, 0]]
            # and flip x axis
            pc[:, 2] = -pc[:, 2]

            pc.unsqueeze_(0)

            logits = pointclip_zs.model_inference(pc[..., :3], label)
            predicted_label = logits.argmax().item()
            labels.append(label)
            preds.append(predicted_label)

            if counter % 1000 == 0:
                print("Processed {} samples".format(counter))
                print_stats(labels, preds)

    print("Finished processing all samples")
    print("Processed {} samples".format(counter))

    if len(args.pre_computed_feat_path) > 0:
        print(f"Keep {np.sum(mask)} within {DISTANCE_THRESH}m and at least {POINT_THRESH} points")
        mask = torch.tensor(mask).to(pointclip_zs.feat_store.device).to(torch.bool)
        labels = pointclip_zs.label_store[mask].cpu().numpy()

        # compute logits
        text_feat = pointclip_zs.textual_encoder()
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        logit_scale = pointclip_zs.logit_scale.exp()

        image_feat = pointclip_zs.feat_store[mask]
        image_feat.to(text_feat.device)
        logits = logit_scale * image_feat @ text_feat.t() * 1.0
        preds = logits.argmax(dim=-1).cpu().numpy()

    print_stats(labels, preds)

    if len(args.pre_computed_feat_path) > 0:
        return

    label_filename = f"label_store_{args.clip_model.replace('/', '')}.pt"
    feat_filename = f"label_store_{args.clip_model.replace('/', '')}.pt"
    torch.save(pointclip_zs.label_store, os.path.join(args.output_dir, label_filename))
    torch.save(pointclip_zs.feat_store, os.path.join(args.output_dir, feat_filename))


if __name__ == "__main__":
    args = parse_args()
    main(args)
