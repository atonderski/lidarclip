from os.path import join
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

from once_devkit.once import ONCE


CAM_NAMES = ["cam0%d" % cam_num for cam_num in (1, 3, 5, 6, 7, 8, 9)]


class OnceImageLidarDataset(Dataset):
    def __init__(self, data_root: str, img_transform, use_grayscale: bool = False):
        super().__init__()
        self._data_root = join(data_root, "data")
        self._devkit = ONCE(data_root)
        self._frames = self._setup()
        self._img_transform = img_transform
        self._use_grayscale = use_grayscale

    def _setup(self) -> List[Tuple[str, str, str, Dict]]:
        mega_sequence_dict = {
            **self._devkit.val_info,
            **self._devkit.train_info,
            **self._devkit.raw_small_info,
            **self._devkit.raw_medium_info,
            # **self._devkit.raw_large_info,
        }
        self._cam_to_idx = {}
        self._idx_to_cam = {}
        for i, cam in enumerate(self._devkit.camera_names):
            self._cam_to_idx[cam] = i
            self._idx_to_cam[i] = cam

        self._sequence_map = {}
        for i, sequence_id in enumerate(mega_sequence_dict.keys()):
            self._sequence_map[sequence_id] = i

        frames = []
        self._cam_to_velos = np.zeros(
            (len(mega_sequence_dict), len(self._devkit.camera_names), 4, 4)
        )
        self._cam_intrinsics = np.zeros(
            (len(mega_sequence_dict), len(self._devkit.camera_names), 3, 3)
        )
        self._cam_distortions = np.zeros(
            (len(mega_sequence_dict), len(self._devkit.camera_names), 5)
        )
        for sequence_id, seq_info in mega_sequence_dict.items():
            for frame_id, frame_info in seq_info.items():
                # seq_info also stores a list with all frames
                if frame_id == "frame_list":
                    continue
                # frame value (not used) has 'pose', 'calib', 'annos'
                for cam_name in self._devkit.camera_names:
                    frames.append((int(sequence_id), int(frame_id), self._cam_to_idx[cam_name]))

            for cam_name in self._devkit.camera_names:
                seq_idx = self._sequence_map[sequence_id]
                cam_idx = self._cam_to_idx[cam_name]
                self._cam_to_velos[seq_idx, cam_idx] = seq_info[seq_info["frame_list"][0]]["calib"][
                    cam_name
                ]["cam_to_velo"]
                self._cam_intrinsics[seq_idx, cam_idx] = seq_info[seq_info["frame_list"][0]][
                    "calib"
                ][cam_name]["cam_intrinsic"]
                self._cam_distortions[seq_idx, cam_idx] = seq_info[seq_info["frame_list"][0]][
                    "calib"
                ][cam_name]["distortion"]

        self._devkit.val_info = None
        self._devkit.train_info = None
        self._devkit.test_info = None
        self._devkit.raw_small_info = None
        self._devkit.raw_medium_info = None
        self._devkit.raw_large_info = None

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
            image = self._load_image(self._devkit.data_root, sequence_id, frame_id, cam_name)
        except:
            print(f"Failed to load image {sequence_id}/{frame_id}/{cam_name}")
            return self.__getitem__(np.random.randint(0, len(self._frames)))
        # image = to_pil_image(image)
        if self._use_grayscale:
            image = ImageOps.grayscale(image)
        og_size = image.size
        image = self._img_transform(image)
        new_size = image.shape[1:]
        try:
            point_cloud = self._devkit.load_point_cloud(sequence_id, frame_id)
        except:
            print(f"Failed to load point cloud {sequence_id}/{frame_id}/{cam_name}")
            return self.__getitem__(np.random.randint(0, len(self._frames)))
        calib = frame_info["calib"]
        point_cloud = self._transform_lidar_to_cam(point_cloud, calib)
        point_cloud = self._remove_points_outside_cam(point_cloud, og_size, new_size, calib)
        point_cloud = torch.from_numpy(point_cloud).float()

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


def build_loader(datadir, clip_preprocess, batch_size=32, num_workers=16, use_grayscale=False):
    dataset = OnceImageLidarDataset(
        datadir, img_transform=clip_preprocess, use_grayscale=use_grayscale
    )
    loader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=_collate_fn,
        pin_memory=False,
        shuffle=True,
    )
    return loader


def demo_dataset():
    import matplotlib.pyplot as plt
    from einops import rearrange

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
