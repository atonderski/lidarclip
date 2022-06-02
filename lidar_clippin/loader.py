from os.path import join
from typing import Dict, List, Tuple

import cv2
import numpy as np
from once_devkit.once import ONCE

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image


CAM_NAMES = ["cam0%d" % cam_num for cam_num in (1, 3, 5, 6, 7, 8, 9)]


class OnceImageLidarDataset(Dataset):
    def __init__(self, data_root: str, img_transform):
        super().__init__()
        self._data_root = join(data_root, "data")
        self._devkit = ONCE(data_root)
        self._frames = self._setup()
        self._img_transform = img_transform

    def _setup(self) -> List[Tuple[str, str, str, Dict]]:
        mega_sequence_dict = {
            **self._devkit.val_info,
            **self._devkit.train_info,
            **self._devkit.raw_small_info,
            **self._devkit.raw_medium_info,
            # **self._devkit.raw_large_info,
        }
        frames = []
        for sequence_id, seq_info in mega_sequence_dict.items():
            for frame_id, frame_info in seq_info.items():
                # seq_info also stores a list with all frames
                if frame_id == "frame_list":
                    continue
                # frame value (not used) has 'pose', 'calib', 'annos'
                for cam_name in self._devkit.camera_names:
                    frames.append((sequence_id, frame_id, cam_name, frame_info))
        print(f"[Dataset] Found {len(frames)} frames.")
        return frames

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, index):
        """Load image and point cloud.

        The point cloud undergoes the following:
        - transformed to camera
        - all points with negative z-coords (behind camera plane) are removed
        - coordinate system is converted to KITTI-style x-forward, y-left, z-up

        """
        sequence_id, frame_id, cam_name, frame_info = self._frames[index]
        try:
            image = self._devkit.load_image(sequence_id, frame_id, cam_name)
        except:
            return self.__getitem__(np.random.randint(0, len(self._frames)))
        image = to_pil_image(image)
        og_size = image.size
        image = self._img_transform(image)
        new_size = image.shape[1:]

        point_cloud = self._devkit.load_point_cloud(sequence_id, frame_id)
        calib = frame_info["calib"][cam_name]
        point_cloud = self._transform_lidar_to_cam(point_cloud, calib)
        point_cloud = self._remove_points_outside_cam(point_cloud, og_size, new_size, calib)
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

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
    def _remove_points_outside_cam(points_cam, og_size, new_size, cam_calib):
        w_og, h_og = og_size
        w_new, h_new = new_size
        new_cam_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
            cam_calib["cam_intrinsic"],
            cam_calib["distortion"],
            (w_og, h_og),
            alpha=1.0,
            newImgSize=(w_new, h_new),
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
        w_ok = np.bitwise_and(0 < points_img[:, 0], points_img[:, 0] < w_new)
        h_ok = np.bitwise_and(0 < points_img[:, 1], points_img[:, 1] < h_new)
        mask = np.bitwise_and(w_ok, h_ok)

        return points_cam[mask]


def _collate_fn(batch):
    batched_img = default_collate([elem[0] for elem in batch])
    batched_pc = [elem[1] for elem in batch]
    return batched_img, batched_pc


def build_loader(datadir, clip_preprocess, batch_size=32, num_workers=16):
    dataset = OnceImageLidarDataset(datadir, img_transform=clip_preprocess)
    loader = DataLoader(
        dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=_collate_fn
    )
    return loader


def demo_dataset():
    import matplotlib.pyplot as plt
    from einops import rearrange

    datadir = "/home/s0001396/Documents/phd/datasets/once"
    datadir = "/Users/s0000960/data/once"
    loader = build_loader(datadir, ToTensor(), num_workers=0, batch_size=2)
    images, lidars = next(iter(loader))

    image = rearrange(images[0], "c h w -> h w c")
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
