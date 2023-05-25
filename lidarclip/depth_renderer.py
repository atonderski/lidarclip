from typing import Tuple
import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
)
from torchvision.transforms import Normalize

# from kitti (forward, left, up) to torch3d (left, up, forward)
KITTI_TO_TORCH3D = (1, 2, 0)


class DepthRenderer:
    def __init__(
        self, aug: bool = False, img_size: int = 224, radius: float = 0.02, aug_scale: float = 5.0
    ):
        self.aug = aug
        self._normalize = Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )
        self._aug_scale = aug_scale
        self._settings = PointsRasterizationSettings(
            image_size=img_size, radius=radius, points_per_pixel=1, bin_size=0
        )

    def __call__(self, point_cloud: torch.Tensor, intrinsics: torch.Tensor, img_size: Tuple):
        batch_size = point_cloud.shape[0]
        cam_t = torch.zeros((batch_size, 3), device=point_cloud.device)
        cam_r = torch.eye(3, device=point_cloud.device).unsqueeze(0).repeat(batch_size, 1, 1)
        focal_length = torch.stack(
            [intrinsics[:, 0, 0], intrinsics[:, 1, 1]],
            dim=-1,
        )
        principal_point = torch.stack(
            [intrinsics[:, 0, 2], intrinsics[:, 1, 2]],
            dim=-1,
        )
        if self.aug:
            # randomly shift +- aug-scale in z direction
            cam_t[:, 2] = self._aug_scale * (
                1 - 2 * torch.rand(batch_size, device=point_cloud.device)
            )
        cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            in_ndc=False,
            image_size=(img_size[::-1],),
            T=cam_t,
            R=cam_r,
            device=point_cloud.device,
        )
        self._settings.radius = float(self._settings.radius)
        renderer = PointsRasterizer(cameras=cameras, raster_settings=self._settings).to(
            point_cloud.device
        )
        point_cloud = point_cloud[..., KITTI_TO_TORCH3D].to(torch.float32)
        pc = Pointclouds(points=point_cloud).to(point_cloud.device)
        return self._norm(torch.mean(renderer(pc).zbuf, dim=-1)).detach()

    def _norm(self, img):  # [B, H, W]
        detached_img = img.detach()
        B, H, W = detached_img.shape

        mask = detached_img > 0
        batch_points = detached_img.reshape(B, -1)
        batch_max, _ = torch.max(batch_points, dim=1, keepdim=True)
        batch_max = batch_max.unsqueeze(-1).repeat(1, H, W)
        detached_img[~mask] = 1.0
        batch_points = detached_img.reshape(B, -1)
        batch_min, _ = torch.min(batch_points, dim=1, keepdim=True)
        batch_min = batch_min.unsqueeze(-1).repeat(1, H, W)
        img = img.sub_(batch_min).div_(batch_max) * 200.0 / 255.0
        img[~mask] = 1.0
        return self._normalize(img.unsqueeze(1).repeat(1, 3, 1, 1))
