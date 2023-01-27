# from https://github.com/ZrrSkywalker/PointCLIP/blob/main/trainers/search_weights.py
import argparse
import os.path as osp

import clip
import torch

from lidarclip.anno_loader import CLASSES

CUSTOM_TEMPLATES_ZS = {
    "ONCE": "point cloud depth map of a {}.",
}

CUSTOM_TEMPLATES_FS = {
    "ONCE": "point cloud of a big {}.",
}


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def textual_encoder(args, classnames, templates, clip_model):
    temp = templates[args.dataset_name]
    prompts = [temp.format(c.replace("_", " ")) for c in classnames]
    prompts = torch.cat([clip.tokenize(p) for p in prompts])
    prompts = prompts.cuda()
    text_feat = clip_model.encode_text(prompts).repeat(1, args.num_views)
    return text_feat


def format_output(acc, a, b, c, d, e, f):
    s = "New best accuracy: {:.2f}, view weights: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}"
    return s.format(acc, a, b, c, d, e, f)


@torch.no_grad()
def search_weights_zs(args):

    print("\n***** Searching for view weights *****")

    image_feat = torch.load(
        osp.join(args.input_dir, f"feat_store_{args.clip_model.replace('/', '')}.pt")
    )
    image_feat = torch.cat(image_feat, dim=0).cuda()
    labels = torch.load(
        osp.join(args.input_dir, f"label_store_{args.clip_model.replace('/', '')}.pt")
    )
    labels = torch.tensor(labels).cuda()

    clip_model, _ = clip.load(args.clip_model, device="cuda")
    clip_model.eval()

    text_feat = textual_encoder(args, CLASSES, CUSTOM_TEMPLATES_ZS, clip_model)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    # Before search
    logits = clip_model.logit_scale.exp() * image_feat @ text_feat.t() * 1.0
    acc, _ = accuracy(logits, labels, topk=(1, 5))

    acc = (acc / image_feat.shape[0]) * 100
    print(f"=> Before search, PointCLIP accuracy: {acc:.2f}")

    # Search
    print("Start to search:")

    best_acc = 0
    # Search_time can be modulated in the config for faster search
    search_time, search_range = args.search_time, args.search_range
    search_list = [(i + 1) * search_range / search_time for i in range(search_time)]

    for a in search_list:
        for b in search_list:
            for c in search_list:
                for d in search_list:
                    for e in search_list:
                        for f in search_list:
                            # Reweight different views
                            view_weights = torch.tensor([a, b, c, d, e, f]).cuda()
                            image_feat_w = image_feat.reshape(
                                -1, args.num_views, clip_model.visual.output_dim
                            ) * view_weights.reshape(1, -1, 1)
                            image_feat_w = image_feat_w.reshape(
                                -1, args.num_views * clip_model.visual.output_dim
                            ).type(clip_model.dtype)

                            logits = (
                                clip_model.logit_scale.exp() * image_feat_w @ text_feat.t() * 1.0
                            )
                            acc, _ = accuracy(logits, labels, topk=(1, 5))
                            acc = (acc / image_feat.shape[0]) * 100

                            if acc > best_acc:
                                print(format_output(acc, a, b, c, d, e, f))
                                best_acc = acc

    print(f"=> After search, PointCLIP accuracy: {best_acc:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="PointCLIP")
    parser.add_argument(
        "--input-dir", type=str, default="/mimer/NOBACKUP/groups/clippin/datasets/once/cls"
    )
    parser.add_argument(
        "--output-dir", type=str, default="/mimer/NOBACKUP/groups/clippin/datasets/once/cls"
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-L/14",
        help="CLIP model",
        choices=["ViT-B/32", "RN50", "RN101", "RN50x4", "RN50x16", "ViT-L/14", "ViT-B/16"],
    )
    parser.add_argument("--dataset_name", type=str, default="ONCE")
    parser.add_argument(
        "--num_views",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--search-time",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--search-range",
        type=int,
        default=1,
    )

    return parser.parse_args()


def main(args):
    search_weights_zs(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
