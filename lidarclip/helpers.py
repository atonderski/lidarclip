import itertools
import os

import numpy as np

import torch

import clip


def logit_img_txt(img_feat, txt_feat, clip_model):
    img_feat = img_feat / img_feat.norm(dim=1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = clip_model.logit_scale.exp().float()
    logits_per_image = logit_scale * img_feat.float() @ txt_feat.t().float()
    logits_per_text = logits_per_image.t()
    return logits_per_text, logits_per_image


def get_topk(prompts, k, img_feats, lidar_feats, joint_feats, clip_model, device):
    text = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
        text_features = text_features.sum(axis=0, keepdim=True)
    logits_per_text_i, _ = logit_img_txt(img_feats, text_features, clip_model)
    logits_per_text_l, _ = logit_img_txt(lidar_feats, text_features, clip_model)
    # Note: Here we use feature averaging, one could instead use score averaging or a re-ranking approach
    logits_per_text_j, _ = logit_img_txt(joint_feats, text_features, clip_model)

    _, img_idxs = torch.topk(logits_per_text_i[0, :], k)
    _, pc_idxs = torch.topk(logits_per_text_l[0, :], k)
    _, joint_idxs = torch.topk(logits_per_text_j[0, :], k)

    return img_idxs.cpu().numpy(), pc_idxs.cpu().numpy(), joint_idxs.cpu().numpy()


def get_topk_separate_prompts(
    image_prompts, lidar_prompts, k, img_feats, lidar_feats, clip_model, device
):
    with torch.no_grad():
        text_features_image = clip_model.encode_text(clip.tokenize(image_prompts).to(device)).sum(
            axis=0, keepdim=True
        )
        text_features_lidar = clip_model.encode_text(clip.tokenize(lidar_prompts).to(device)).sum(
            axis=0, keepdim=True
        )
    logits_per_text_i, _ = logit_img_txt(img_feats, text_features_image, clip_model)
    logits_per_text_l, _ = logit_img_txt(lidar_feats, text_features_lidar, clip_model)
    # Note: Here we use score averaging, but one could instead use a re-ranking approach
    logits_per_text_j = logits_per_text_i + logits_per_text_l

    _, pc_idxs = torch.topk(logits_per_text_l[0, :], k)
    _, img_idxs = torch.topk(logits_per_text_i[0, :], k)
    _, joint_idxs = torch.topk(logits_per_text_j[0, :], k)

    return img_idxs.cpu().numpy(), pc_idxs.cpu().numpy(), joint_idxs.cpu().numpy()


def filter_candidates(candidate_idxs, k, exclude_range):
    """Filter out candidates that are too close to each other."""
    idxs = []
    exclude_idxs = set()
    for idx in candidate_idxs:
        if idx in exclude_idxs:
            continue
        exclude_idxs.update(range(idx - exclude_range, idx + exclude_range + 1))
        idxs.append(idx)
        if len(idxs) == k:
            break
    return np.array(idxs)


def try_paths(*paths):
    """Try all directory paths and return the first one that works."""
    for path in paths:
        if os.path.exists(path):
            return path
    raise ValueError("No valid path found in {}".format(paths))


class MultiLoader:
    """Helper for loading data from a list of dataloaders."""

    def __init__(self, loaders):
        self.loaders = loaders
        self.dataset_lens = [len(loader.dataset) for loader in loaders]
        self.bins = np.cumsum(self.dataset_lens).tolist()
        assert len(self.loaders) == len(self.dataset_lens)
        # self.dataset = self

    def __getitem__(self, idx):
        # Which loader should we look at?
        dataset_idx = np.digitize(idx, self.bins)
        assert dataset_idx < len(self.loaders), f"dataset_idx is {dataset_idx}, idx is {idx}"
        idx_to_subtract = self.bins[dataset_idx]

        idx -= idx_to_subtract
        if idx > len(self):
            raise IndexError(f"idx is {idx}, len is {len(self)}")

        return self.loaders[dataset_idx].dataset[idx]

    def __len__(self):
        return sum(self.dataset_lens)

    def __iter__(self):
        return itertools.chain(*self.loaders)
