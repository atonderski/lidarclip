import argparse
import os

import numpy as np
from pointclip_utils import PointCLIP_ZS

import torch

import clip

from lidarclip.anno_loader import CLASSES


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
        default="/home/s0001396/Documents/phd/datasets/once/cls/val.pt",
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
    args = parser.parse_args()
    return args


def print_stats(labels, preds):
    # print accuracy
    print("Accuracy: ", np.sum(np.array(labels) == np.array(preds)) / len(labels))
    # print class-wise accuracy
    print("Class-wise accuracy:")
    for c in CLASSES:
        mask = np.array(labels) == c

        print(c, np.sum(np.array(labels)[mask] == np.array(preds)[mask]) / np.sum(mask))

    # print confusion matrix using scikit-learn
    print("Confusion matrix:")
    from sklearn.metrics import confusion_matrix

    print(confusion_matrix(labels, preds, labels=CLASSES))


def main(args):
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # load data
    data = torch.load(args.data_dir)
    # load model
    clip_model, clip_preprocess = clip.load(args.clip_model, device="cpu")

    pointclip_zs = PointCLIP_ZS()
    pointclip_zs.build_model(clip_model, CLASSES, args.num_views)
    labels = []
    preds = []

    counter = 0
    for i in range(len(data)):
        anno = data[i]
        for j in range(len(anno["names"])):
            counter += 1

            label = CLASSES.index(anno["names"][j])
            pc = anno["points_per_obj"][j]
            x, y, z, length, w, h, ry = anno["boxes_3d"][j]
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

            if counter % 100 == 0:
                print("Processed {} samples".format(counter))
                print_stats(labels, preds)

    print("Finished processing all samples")
    print("Processed {} samples".format(counter))
    print_stats(labels, preds)

    torch.save(pointclip_zs.label_store, os.path.join(args.output_dir, "label_store.pt"))
    torch.save(pointclip_zs.feat_store, os.path.join(args.output_dir, "feat_store.pt"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
