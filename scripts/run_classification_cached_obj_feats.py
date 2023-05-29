import argparse
import sys
from collections import defaultdict
from typing import Dict

import clip
import torch

sys.path.append("..")
from lidarclip.anno_loader import CLASSES
from lidarclip.prompts import OBJECT_PROMPT_TEMPLATES
from lidarclip.helpers import logit_img_txt, try_paths

device = "cuda" if torch.cuda.is_available() else "cpu"


CLIP_VERSION = "ViT-L/14"
# CLIP_VERSION = "ViT-B/32"
# CLIP_VERSION = "RN101"


def gen_cls_embedding(cls_name: str, clip_model) -> torch.Tensor:
    print(f"Generating embedding for {cls_name}")
    prompts = [template.format(cls_name) for template in OBJECT_PROMPT_TEMPLATES]
    # if cls_name == "Car":
    #     cls_name = "Vehicle"
    # prompts = [f"A photo of a {cls_name} on the street or in the city"]
    with torch.no_grad():
        tokenized_prompts = clip.tokenize(prompts).to(device)
        cls_features = clip_model.encode_text(tokenized_prompts)
        return cls_features.sum(axis=0, keepdim=True)


def compute_accuracy(
    obj_feats: torch.Tensor, tru_class_idx: int, cls_embeddings_pt, clip_model
) -> Dict[str, float]:
    logits_per_text, _ = logit_img_txt(obj_feats, cls_embeddings_pt, clip_model)
    score_per_class = logits_per_text.softmax(0).T
    accuracies = {}
    for k in range(1, min(6, score_per_class.shape[1] + 1)):
        topk = (
            score_per_class.argsort(axis=1, descending=True)[:, :k] == tru_class_idx
        ).sum() / len(score_per_class)
        accuracies[f"top-{k}"] = topk
    return accuracies


def main(args):
    # Load data and features
    clip_model, clip_preprocess = clip.load(args.clip_version, device=device)
    feature_root = try_paths(
        "/proj/nlp4adas/features", "../features", "/mimer/NOBACKUP/groups/clippin/features"
    )
    obj_feats = torch.load(f"{feature_root}/{args.features_filename}", map_location=device)
    for class_name, cls_feats in obj_feats.items():
        print(class_name, len(cls_feats))
        obj_feats[class_name] = torch.stack(cls_feats)

    ##############################################################

    CATEGORIES = CLASSES

    cls_embeddings = {name: gen_cls_embedding(name, clip_model) for name in CATEGORIES}
    print("Generated embeddings for: ", list(cls_embeddings.keys()))
    cls_embeddings_pt = torch.vstack(list(cls_embeddings.values()))

    ##############################################################

    overall_clsavg = defaultdict(float)
    overall_objavg = defaultdict(float)
    for class_name, cls_obj_feats in obj_feats.items():
        if class_name not in CATEGORIES:
            continue
        print("Evaluating class", class_name, f"(n={len(cls_obj_feats)})")
        accuracies = compute_accuracy(
            cls_obj_feats, CATEGORIES.index(class_name), cls_embeddings_pt, clip_model
        )
        res_string = ", ".join(f"{k}: {v:.3f} ({v*100:.1f}%)" for k, v in accuracies.items())
        print(f"  {res_string}")
        for k, v in accuracies.items():
            overall_clsavg[k] += v
            overall_objavg[k] += v * len(cls_obj_feats)

    print("\nOverall (cls avg):")
    overall_clsavg = {k: v / len(CATEGORIES) for k, v in overall_clsavg.items()}
    res_string = ", ".join(f"{k}: {v:.3f} ({v*100:.1f}%)" for k, v in overall_clsavg.items())
    print(f"  {res_string}")

    print("\nOverall (obj avg)")
    num_objs = sum(len(v) for v in obj_feats.values())
    overall_objavg = {k: v / num_objs for k, v in overall_objavg.items()}
    res_string = ", ".join(f"{k}: {v:.3f} ({v*100:.1f}%)" for k, v in overall_objavg.items())
    print(f"  {res_string}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-version", type=str, default="ViT-L/14")
    parser.add_argument("--features_filename", type=str, default="")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
