import json
import os
from copy import deepcopy

import numpy as np
from PIL import ImageOps

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from lidar_clippin.loader import CAM_NAMES, OnceImageLidarDataset


CENTERCROP_BOX = [450, 0, 1470, 1020]


class OnceFullDataset(OnceImageLidarDataset):
    def __init__(
        self, data_root: str, img_transform, use_grayscale: bool = False, split: str = "val"
    ):
        assert split in (
            "train-only",
            "val",
        ), "Annotations are only available for train and val splits."
        super().__init__(
            data_root=data_root,
            img_transform=img_transform,
            use_grayscale=use_grayscale,
            split=split,
        )
        self._setup_for_annos()

    def _setup_for_annos(self):
        """Setup annotations for all frames."""
        frames = []
        annos = {}
        for sequence_id in self._sequence_map:
            # Load annotation file for sequence
            annos[sequence_id] = {}
            anno_file_path = os.path.join(
                self._data_root, sequence_id, "{}.json".format(sequence_id)
            )
            with open(anno_file_path, "r") as f:
                seq_anno = json.load(f)
            for frame_anno in seq_anno["frames"]:
                frame_id = frame_anno["frame_id"]
                if "annos" in frame_anno:
                    frames.append((int(sequence_id), int(frame_id)))
                    annos[sequence_id][frame_id] = frame_anno["annos"]
        print(f"[Dataset] Kept {len(frames)*len(CAM_NAMES)} frames that have annotations.")
        # Override existing frames with frames that actually have annotations
        self._frames = torch.as_tensor(frames)
        self._annos = annos

    def __getitem__(self, index):
        """Load image, point cloud, and annotations.

        The point cloud undergoes the following:
        - transformed to camera
        - all points with negative z-coords (behind camera plane) are removed
        - coordinate system is converted to KITTI-style x-forward, y-left, z-up

        """
        sequence_id, frame_id, cam_idx, seq_idx, cam_name = self.map_index(index)
        image = self._load_image(self._data_root, sequence_id, frame_id, cam_name)
        if self._use_grayscale:
            image = ImageOps.grayscale(image)
        og_size = image.size
        image = self._img_transform(image)
        new_size = image.shape[1:]
        point_cloud = self._load_point_cloud(self._data_root, sequence_id, frame_id)
        calib = {
            "cam_to_velo": self._cam_to_velos[seq_idx, cam_idx],
            "cam_intrinsic": self._cam_intrinsics[seq_idx, cam_idx],
            "distortion": self._cam_distortions[seq_idx, cam_idx],
        }
        point_cloud = self._transform_lidar_and_remove_points_outside_cam_torch(
            point_cloud, calib, og_size, new_size
        )

        annos = deepcopy(self._annos[sequence_id][frame_id])
        annos["boxes_2d"] = annos["boxes_2d"][cam_name]
        annos = self._keep_annos_in_image(annos)
        return image, point_cloud, annos

    def _keep_annos_in_image(self, annos):
        # Check if any part of the bounding box is within the CENTERCROP_BOX (xyxy format)
        boxes = np.array(annos["boxes_2d"])
        mask = (
            (boxes[:, 2] >= CENTERCROP_BOX[0])
            & (boxes[:, 0] <= CENTERCROP_BOX[2])
            & (boxes[:, 3] >= CENTERCROP_BOX[1])
            & (boxes[:, 1] <= CENTERCROP_BOX[3])
        )
        new_anno = {"names": [], "boxes_2d": [], "boxes_3d": []}
        for name, box_2d, box_3d, mask in zip(
            annos["names"], annos["boxes_2d"], annos["boxes_3d"], mask
        ):
            if mask:
                new_anno["names"].append(name)
                new_anno["boxes_2d"].append(box_2d)
                new_anno["boxes_3d"].append(box_3d)
        return new_anno


def _collate_fn(batch):
    batched_img = default_collate([elem[0] for elem in batch])
    batched_pc = [elem[1] for elem in batch]
    batched_annos = [elem[2] for elem in batch]
    return batched_img, batched_pc, batched_annos


def build_loader(
    datadir,
    clip_preprocess,
    batch_size=32,
    num_workers=16,
    use_grayscale=False,
    split="val",
    shuffle=False,
):
    dataset = OnceFullDataset(
        datadir, img_transform=clip_preprocess, use_grayscale=use_grayscale, split=split
    )
    loader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=_collate_fn,
        pin_memory=False,
        shuffle=shuffle,
    )
    return loader


def demo_dataset():
    import matplotlib.pyplot as plt
    from einops import rearrange

    import torch

    import clip

    _, clip_preprocess = clip.load("ViT-B/32")

    # datadir = "/home/s0001396/Documents/phd/datasets/once"
    datadir = "/Users/s0000960/data/once"
    loader = build_loader(datadir, clip_preprocess, num_workers=0, batch_size=2, split="val")
    images, lidars, annos = next(iter(loader))

    means = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cpu")
    stds = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cpu")
    image = rearrange(images[0], "c h w -> h w c") * stds + means

    lidar = lidars[0]
    lidar = lidar[torch.randperm(lidar.shape[0])[:8192]]
    plt.figure()
    plt.imshow(image)
    plt.figure()
    # for visualization convert to x-right, y-forward
    plt.scatter(-lidar[:, 1], lidar[:, 0], s=0.1, c=np.clip(lidar[:, 3], 0, 1), cmap="coolwarm")
    plt.axis("equal")
    plt.xlim(-10, 10)
    plt.ylim(0, 40)
    plt.show()


if __name__ == "__main__":
    demo_dataset()
