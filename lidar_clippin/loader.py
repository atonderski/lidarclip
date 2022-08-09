import gc
import json
import os
from os.path import join
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from torch import from_numpy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate


CAM_NAMES = ["cam0%d" % cam_num for cam_num in (1, 3, 5, 6, 7, 8, 9)]
SPLITS = {
    "train": ("train", "raw_small", "raw_medium", "raw_large"),
    "val": ("val",),
    "test": ("test",),
}


class OnceImageLidarDataset(Dataset):
    def __init__(
        self, data_root: str, img_transform, use_grayscale: bool = False, split: str = "train"
    ):
        super().__init__()
        self._data_root = join(data_root, "data")
        self._frames = self._setup(split)
        self._img_transform = img_transform
        self._use_grayscale = use_grayscale
        gc.collect()

    def _setup(self, split: str) -> List[Tuple[str, str, str, Dict]]:
        assert split in SPLITS, f"Unknown split: {split}, must be one of {SPLITS.keys()}"

        seq_list = []
        for attr in SPLITS[split]:
            seq_list_path = os.path.join(self._data_root, "..", "ImageSets", f"{attr}.txt")
            if not os.path.exists(seq_list_path):
                continue
            with open(seq_list_path, "r") as f:
                seq_list.extend(set(map(lambda x: x.strip(), f.readlines())))

        seq_list = sorted(seq_list)
        self._sequence_map = {seq: i for i, seq in enumerate(seq_list)}

        self._cam_to_idx = {}
        self._idx_to_cam = {}
        for i, cam in enumerate(CAM_NAMES):
            self._cam_to_idx[cam] = i
            self._idx_to_cam[i] = cam

        frames = []
        self._cam_to_velos = np.zeros((len(seq_list), len(CAM_NAMES), 4, 4))
        self._cam_intrinsics = np.zeros((len(seq_list), len(CAM_NAMES), 3, 3))
        self._cam_distortions = np.zeros((len(seq_list), len(CAM_NAMES), 5))
        for sequence_id in seq_list:
            anno_file_path = os.path.join(
                self._data_root, sequence_id, "{}.json".format(sequence_id)
            )
            if not os.path.isfile(anno_file_path):
                print("no annotation file for sequence {}".format(sequence_id))
                raise FileNotFoundError

            with open(anno_file_path, "r") as f:
                anno = json.load(f)

            for frame_anno in anno["frames"]:
                frame_id = frame_anno["frame_id"]
                # frame value (not used) has 'pose', 'calib', 'annos'
                for cam_name in CAM_NAMES:
                    frames.append((int(sequence_id), int(frame_id), self._cam_to_idx[cam_name]))

            for cam_name in CAM_NAMES:
                seq_idx = self._sequence_map[sequence_id]
                cam_idx = self._cam_to_idx[cam_name]
                self._cam_to_velos[seq_idx, cam_idx] = np.array(
                    anno["calib"][cam_name]["cam_to_velo"]
                )
                self._cam_intrinsics[seq_idx, cam_idx] = np.array(
                    anno["calib"][cam_name]["cam_intrinsic"]
                )
                self._cam_distortions[seq_idx, cam_idx] = np.array(
                    anno["calib"][cam_name]["distortion"]
                )

        print(f"[Dataset] Found {len(frames)} frames.")
        return np.array(frames)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, index):
        """Load image and point cloud.

        The point cloud undergoes the following:
        - transformed to camera
        - all points with negative z-coords (behind camera plane) are removed
        - coordinate system is converted to KITTI-style x-forward, y-left, z-up

        """
        sequence_id, frame_id, cam_idx = self._frames[index]
        sequence_id = str(sequence_id).zfill(6)
        seq_idx = self._sequence_map[sequence_id]
        frame_info = {
            "calib": {
                "cam_to_velo": self._cam_to_velos[seq_idx, cam_idx],
                "cam_intrinsic": self._cam_intrinsics[seq_idx, cam_idx],
                "distortion": self._cam_distortions[seq_idx, cam_idx],
            }
        }
        frame_id = str(frame_id)
        cam_name = self._idx_to_cam[cam_idx]
        try:
            image = self._load_image(self._data_root, sequence_id, frame_id, cam_name)
        except:
            print(f"Failed to load image {sequence_id}/{frame_id}/{cam_name}")
            # return self.__getitem__(np.random.randint(0, len(self._frames)))
        # image = to_pil_image(image)
        if self._use_grayscale:
            image = ImageOps.grayscale(image)
        og_size = image.size
        image = self._img_transform(image)
        new_size = image.shape[1:]
        try:
            point_cloud = self._load_point_cloud(self._data_root, sequence_id, frame_id)
        except:
            print(f"Failed to load point cloud {sequence_id}/{frame_id}/{cam_name}")
            # return self.__getitem__(np.random.randint(0, len(self._frames)))
        calib = frame_info["calib"]
        point_cloud = self._transform_lidar_to_cam(point_cloud, calib)
        point_cloud = self._remove_points_outside_cam(point_cloud, og_size, new_size, calib).copy()
        point_cloud = from_numpy(point_cloud).float()

        return image, point_cloud

    @staticmethod
    def _transform_lidar_to_cam(points_lidar, calibration):
        cam_2_lidar = calibration["cam_to_velo"]
        point_xyz = points_lidar[:, :3]
        points_homo = np.hstack(
            [
                points_lidar[:, :3],
                np.ones(point_xyz.shape[0], dtype=np.float32).reshape((-1, 1)),
            ]
        )
        points_cam = np.dot(points_homo, np.linalg.inv(cam_2_lidar).T)
        mask = points_cam[:, 2] > 0
        points_cam = points_cam[mask]  # discard points behind camera
        # Convert from openc camera coordinates to KITTI style (x-forward, y-left, z-up)
        point_cam_with_reflectance = np.hstack(
            [
                points_cam[:, 2:3],  # z -> x
                -points_cam[:, 0:1],  # -x -> y
                -points_cam[:, 1:2],  # -y -> z
                points_lidar[mask][:, 3:],  # add original reflectance
            ]
        )
        return point_cam_with_reflectance

    @staticmethod
    def _load_image(data_root, seq_id, frame_id, cam_name):
        img_path = join(data_root, seq_id, cam_name, "{}.jpg".format(frame_id))
        return Image.open(img_path)

    @staticmethod
    def _load_point_cloud(data_root, seq_id, frame_id):
        bin_path = join(data_root, seq_id, "lidar_roof", "{}.bin".format(frame_id))
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points

    @staticmethod
    def _remove_points_outside_cam(points_cam, og_size, new_size, cam_calib):
        w_og, h_og = og_size
        og_short_side = min(w_og, h_og)
        w_new, h_new = new_size
        scaling = og_short_side / w_new
        new_cam_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
            cam_calib["cam_intrinsic"],
            cam_calib["distortion"],
            (w_og, h_og),
            alpha=0.0,
            # newImgSize=(int(w_og//scaling), int(h_og//scaling)),
        )
        cam_intri = np.hstack([new_cam_intrinsic, np.zeros((3, 1), dtype=np.float32)])
        points_cam_img_coors = points_cam[:, [1, 2, 0]]
        points_cam_img_coors[:, 0] = -points_cam_img_coors[:, 0]
        points_cam_img_coors[:, 1] = -points_cam_img_coors[:, 1]

        points_cam_img_coors = np.hstack(
            [
                points_cam_img_coors,
                np.ones(points_cam_img_coors.shape[0], dtype=np.float32).reshape((-1, 1)),
            ]
        )

        points_img = np.dot(points_cam_img_coors, cam_intri.T)
        points_img = points_img / points_img[:, [2]]
        left_border = w_og // 2 - h_og // 2
        right_border = w_og // 2 + h_og // 2
        w_ok = np.bitwise_and(left_border < points_img[:, 0], points_img[:, 0] < right_border)
        h_ok = np.bitwise_and(0 < points_img[:, 1], points_img[:, 1] < h_og)
        mask = np.bitwise_and(w_ok, h_ok)

        return points_cam[mask]


def _collate_fn(batch):
    batched_img = default_collate([elem[0] for elem in batch])
    batched_pc = [elem[1] for elem in batch]
    return batched_img, batched_pc


def build_loader(
    datadir,
    clip_preprocess,
    batch_size=32,
    num_workers=16,
    use_grayscale=False,
    split="train",
    shuffle=False,
):
    dataset = OnceImageLidarDataset(
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

    datadir = "/home/s0001396/Documents/phd/datasets/once"
    # datadir = "/Users/s0000960/data/once"
    loader = build_loader(datadir, clip_preprocess, num_workers=0, batch_size=2)
    images, lidars = next(iter(loader))

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
