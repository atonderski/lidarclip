import json
import os
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from lidarclip.loader import (
    CAM_NAMES,
    NUSCENES_CAM_NAMES,
    NuscenesImageLidarDataset,
    OnceImageLidarDataset,
)

# ONCE settings
CENTERCROP_BOX = [450, 0, 1470, 1020]
CLASSES = ("Car", "Truck", "Bus", "Pedestrian", "Cyclist")
WEATHERS = ("sunny", "cloudy", "rainy")
PERIODS = ("morning", "noon", "afternoon", "night")


NUSCENES_CLASSES = (
    "Car",
    "Truck",
    "Bus",
    "Trailer",
    "Pedestrian",
    "Cyclist",
    # "Animal",
    # "Emergency vehicle",
)

# NuScenes classes
NUSCENES_CLASS_MAP = {
    "animal": None,  # Animal
    "human.pedestrian.adult": "Pedestrian",
    "human.pedestrian.child": "Pedestrian",
    "human.pedestrian.construction_worker": "Pedestrian",
    "human.pedestrian.personal_mobility": "Pedestrian",
    "human.pedestrian.police_officer": "Pedestrian",
    "human.pedestrian.stroller": "Pedestrian",
    "human.pedestrian.wheelchair": "Pedestrian",
    "movable_object.barrier": None,
    "movable_object.debris": None,
    "movable_object.pushable_pullable": None,
    "movable_object.trafficcone": None,
    "static_object.bicycle_rack": None,
    "vehicle.bicycle": "Cyclist",
    "vehicle.bus.bendy": "Bus",
    "vehicle.bus.rigid": "Bus",
    "vehicle.car": "Car",
    "vehicle.construction": "Truck",
    "vehicle.emergency.ambulance": None,  # Emergency vehicle
    "vehicle.emergency.police": None,  # Emergency vehicle
    "vehicle.motorcycle": "Cyclist",
    "vehicle.trailer": "Trailer",
    "vehicle.truck": "Truck",
    "flat.driveable_surface": None,
    "flat.other": None,
    "flat.sidewalk": None,
    "flat.terrain": None,
    "static.manmade": None,
    "static.other": None,
    "static.vegetation": None,
    "vehicle.ego": None,
    "noise": None,
}


class NuscenesFullDataset(NuscenesImageLidarDataset):
    def __init__(
        self,
        data_root: str,
        img_transform,
        split: str = "val",
        min_dist: float = 0.5,
        skip_data: bool = False,
        skip_anno: bool = False,
    ):
        super().__init__(
            data_root=data_root,
            img_transform=img_transform,
            split=split,
            min_dist=min_dist,
        )
        self._skip_anno = skip_anno
        self._skip_data = skip_data

    def _setup(self, split: str) -> List[Tuple[str, str, str]]:
        ok_scene_tokens = self._get_ok_scene_tokens(split)
        return [
            sample["token"]
            for sample in self._nusc.sample
            if sample["scene_token"] in ok_scene_tokens
        ]

    def __getitem__(self, index):
        anno, meta = None, None
        if self._skip_data:
            img, pc = torch.zeros((3, 0, 0)), torch.zeros((0, 4))
        else:
            img, pc = super().__getitem__(index)
        if not self._skip_anno:
            anno, meta = self._get_anno_meta(index)
        return img, pc, anno, meta

    def _get_anno_meta(self, index):
        sample_token = self._frames[index // len(NUSCENES_CAM_NAMES)]
        cam_name = NUSCENES_CAM_NAMES[index % len(NUSCENES_CAM_NAMES)]
        sample = self._nusc.get("sample", sample_token)
        cam_token = sample["data"][cam_name]
        # Note: this filters out boxes that are not visible in the camera
        data_path, boxes, cam_intrinsic = self._nusc.get_sample_data(cam_token)
        # TODO: object coordinate system is in camera frame
        boxes = [box for box in boxes if NUSCENES_CLASS_MAP[box.name]]
        anno = {
            "names": [NUSCENES_CLASS_MAP[box.name] for box in boxes],
            "boxes_2d": [None for _ in boxes],
            "boxes_3d": [
                torch.Tensor((*box.center, *box.wlh, box.orientation.radians)) for box in boxes
            ],
        }
        meta = None  # TODO: implement nuscenes metadata
        return anno, meta


class OnceFullDataset(OnceImageLidarDataset):
    def __init__(
        self,
        data_root: str,
        img_transform,
        split: str = "val",
        skip_data: bool = False,
        skip_anno: bool = False,
        return_points_per_obj: bool = False,
    ):
        assert (
            split
            in (
                "train-only",
                "val",
            )
            or skip_anno
        ), "Annotations are only available for train and val splits."
        assert not (
            return_points_per_obj and (skip_data or skip_anno)
        ), "Cannot skip data/anno if not returning points per object."
        super().__init__(
            data_root=data_root,
            img_transform=img_transform,
            split=split,
        )
        self._skip_anno = skip_anno
        self._skip_data = skip_data
        self._return_points_per_obj = return_points_per_obj
        self._setup_for_annos()

    def _setup_for_annos(self):
        """Setup annotations for all frames."""
        frames = []
        annos = {}
        meta_info = {}
        for sequence_id in self._sequence_map:
            # Load annotation file for sequence
            annos[sequence_id] = {}
            meta_info[sequence_id] = {}
            anno_file_path = os.path.join(
                self._data_root, sequence_id, "{}.json".format(sequence_id)
            )
            with open(anno_file_path, "r") as f:
                seq_anno = json.load(f)
            meta_info[sequence_id]["weather"] = seq_anno["meta_info"]["weather"]
            meta_info[sequence_id]["period"] = seq_anno["meta_info"]["period"]
            for frame_anno in seq_anno["frames"]:
                frame_id = frame_anno["frame_id"]
                if self._skip_anno:
                    frames.append((int(sequence_id), int(frame_id)))
                elif "annos" in frame_anno:
                    frames.append((int(sequence_id), int(frame_id)))
                    annos[sequence_id][frame_id] = frame_anno["annos"]
        print(f"[Dataset] Kept {len(frames)*len(CAM_NAMES)} frames.")
        # Override existing frames with frames that actually have annotations
        self._frames = torch.as_tensor(frames)
        self._annos = annos
        self._meta_info = meta_info

    def __getitem__(self, index):
        """Load image, point cloud, and annotations.

        The point cloud undergoes the following:
        - transformed to camera
        - all points with negative z-coords (behind camera plane) are removed
        - coordinate system is converted to KITTI-style x-forward, y-left, z-up

        """
        sequence_id, frame_id, cam_idx, seq_idx, cam_name = self.map_index(index)
        calib = {
            "cam_to_velo": self._cam_to_velos[seq_idx, cam_idx],
            "cam_intrinsic": self._cam_intrinsics[seq_idx, cam_idx],
            "distortion": self._cam_distortions[seq_idx, cam_idx],
        }

        meta_info = self._meta_info[sequence_id]

        if self._skip_data:
            image, point_cloud = torch.zeros((3, 0, 0)), torch.zeros((0, 4))
        else:
            image = self._load_image(self._data_root, sequence_id, frame_id, cam_name)
            og_size = image.size
            image = self._img_transform(image)
            point_cloud = self._load_point_cloud(self._data_root, sequence_id, frame_id)
            point_cloud = self._transform_lidar_and_remove_points_outside_cam_torch(
                point_cloud, calib, og_size
            )

        if self._skip_anno:
            annos = None
        else:
            annos = deepcopy(self._annos[sequence_id][frame_id])
            annos["boxes_2d"] = annos["boxes_2d"][cam_name]
            annos = self._keep_annos_in_image(annos)
            boxes_3d = torch.tensor(annos["boxes_3d"]).reshape(-1, 7)
            transformed_boxes_center_coord = (
                self._transform_lidar_and_remove_points_outside_cam_torch(
                    boxes_3d[:, :4], calib, og_size=None, remove_points_outside_cam=False
                )
            )
            # Rotation matrix to go from camera to lidar
            rot_mat = calib["cam_to_velo"][:3, :3]
            # Rotation in radians
            rots = boxes_3d[:, -1:]
            # Put rotation as a vector with origo in center of lidar
            rot_points = torch.cat([rots.cos(), rots.sin(), torch.zeros_like(rots)], dim=1)
            # Rotate the rotation vector to camera coordinate system
            rot_points = rot_points @ rot_mat.inverse().T
            # Get the angle of the vector in camera coordinate system
            rots = torch.atan2(rot_points[:, 2:3], rot_points[:, 0:1])
            # Remove pi/2 to get the angle of the box in KITTI format
            rots = rots - np.pi / 2
            transformed_boxes = torch.cat(
                [transformed_boxes_center_coord[:, :3], boxes_3d[:, 3:-1], rots], dim=1
            )
            annos["boxes_3d"] = transformed_boxes
            # x,y,z,l,w,h,rz in kitti format
            # (x,y,z is the center of the box) x is forward, y is left, z is up

            points_per_obj = self._get_points_per_obj(annos, point_cloud)
            annos["points_per_obj"] = points_per_obj
            annos["seq_info"] = {
                "sequence_id": sequence_id,
                "frame_id": frame_id,
                "cam_name": cam_name,
                "seq_idx": seq_idx,
            }

        return image, point_cloud, annos, meta_info

    def _rotate_point_cloud_torch(self, point_cloud, rot_angle):
        """Rotate point cloud by rot_angle around z axis.

        Args:
            point_cloud: (tensor) [N, 4] in ONCE format
            rot_angle: (float) rotation angle in radians

        Returns:
            rotated_point_cloud: (tensor) [N, 4] in ONCE format

        """
        rot_sin = torch.sin(rot_angle)
        rot_cos = torch.cos(rot_angle)
        rotation_matrix = torch.tensor(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
            dtype=point_cloud.dtype,
            device=point_cloud.device,
        )
        # pad rotation matrix to 4x4
        rotation_matrix = torch.cat(
            [
                rotation_matrix,
                torch.zeros((1, 3), dtype=point_cloud.dtype, device=point_cloud.device),
            ],
            dim=0,
        )
        rotation_matrix = torch.cat(
            [
                rotation_matrix,
                torch.zeros((4, 1), dtype=point_cloud.dtype, device=point_cloud.device),
            ],
            dim=1,
        )
        rotation_matrix[3, 3] = 1
        point_cloud = point_cloud @ rotation_matrix
        return point_cloud

    def _get_points_in_box(self, box_3d, point_cloud):
        """Get points inside of 3d box.

        Args:
            box_3d: (tensor) [x, y, z, l, w, h, rz] in ONCE format (left, back, up)
            point_cloud: (tensor) [N, 4] in ONCE format

        Returns:
            points_in_box: (tensor) [N, 4] in ONCE format

        """
        # Transform point cloud to box coordinate system
        box_3d = box_3d.reshape(-1, 7)
        x, y, z, length, w, h, rz = box_3d[0]
        point_cloud = point_cloud - torch.tensor(
            [[x, y, z, 0]], dtype=point_cloud.dtype, device=point_cloud.device
        )
        point_cloud = self._rotate_point_cloud_torch(point_cloud, rz)
        # Get points within box
        mask = (
            (point_cloud[:, 0] >= -length / 2)
            & (point_cloud[:, 0] <= length / 2)
            & (point_cloud[:, 1] >= -w / 2)
            & (point_cloud[:, 1] <= w / 2)
            & (point_cloud[:, 2] >= -h / 2)
            & (point_cloud[:, 2] <= h / 2)
        )
        points_in_box = point_cloud[mask]
        return points_in_box

    def _get_points_per_obj(self, annos, point_cloud):
        boxes_3d = torch.tensor(annos["boxes_3d"]).reshape(-1, 7)
        points_per_obj = []
        for box_3d in boxes_3d:
            box_3d = box_3d.unsqueeze(0)
            points_per_obj.append(self._get_points_in_box(box_3d, point_cloud))
        return points_per_obj

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
    batched_metas = [elem[3] for elem in batch]
    return batched_img, batched_pc, batched_annos, batched_metas


def build_anno_loader(
    datadir,
    clip_preprocess,
    batch_size=32,
    num_workers=16,
    split="val",
    shuffle=False,
    skip_data=False,
    skip_anno=False,
    return_points_per_obj=False,
    dataset_name="once",
):
    if dataset_name == "once":
        dataset = OnceFullDataset(
            datadir,
            img_transform=clip_preprocess,
            split=split,
            skip_data=skip_data,
            skip_anno=skip_anno,
            return_points_per_obj=return_points_per_obj,
        )
    elif dataset_name == "nuscenes":
        dataset = NuscenesFullDataset(
            datadir,
            img_transform=clip_preprocess,
            split=split,
            skip_data=skip_data,
            skip_anno=skip_anno,
        )
    else:
        raise ValueError(f"Unsupported dataset name {dataset_name}")
    loader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=_collate_fn,
        pin_memory=False,
        shuffle=shuffle,
    )
    return loader


def demo_once():
    import clip
    import matplotlib.pyplot as plt
    import torch
    from einops import rearrange

    _, clip_preprocess = clip.load("ViT-B/32")

    # datadir = "/home/s0001396/Documents/phd/datasets/once"
    datadir = "/Users/s0000960/data/once"
    loader = build_anno_loader(datadir, clip_preprocess, num_workers=0, batch_size=2, split="val")
    images, lidars, annos, metas = next(iter(loader))

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


def demo_nuscenes():
    datadir = "/Users/s0000960/data/nuscenes"
    loader = build_anno_loader(
        datadir,
        lambda x: x,
        num_workers=0,
        batch_size=2,
        split="mini",
        skip_data=True,
        dataset_name="nuscenes",
    )
    images, lidars, annos, metas = next(iter(loader))


if __name__ == "__main__":
    # demo_once()
    demo_nuscenes()
