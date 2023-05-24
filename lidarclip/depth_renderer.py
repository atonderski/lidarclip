from typing import Dict
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

    def __call__(self, point_cloud: torch.Tensor, calib: Dict, img_size):
        cam_t = torch.tensor([[0.0, 0.0, 0.0]])
        if self.aug:
            # randomly shift +- aug-scale in z direction
            cam_t[:, 2] = self._aug_scale * (1 - 2 * torch.rand(1))
        cameras = PerspectiveCameras(
            focal_length=((calib["cam_intrinsic"][0, 0], calib["cam_intrinsic"][1, 1]),),
            principal_point=((calib["cam_intrinsic"][0, 2], calib["cam_intrinsic"][1, 2]),),
            in_ndc=False,
            image_size=(img_size[::-1],),
            T=cam_t,
        )
        renderer = PointsRasterizer(cameras=cameras, raster_settings=self._settings)
        pc = Pointclouds(points=point_cloud[None, :, KITTI_TO_TORCH3D])
        return self._norm(torch.mean(renderer(pc).zbuf, dim=-1))

    def _norm(self, img):  # [B, H, W]
        mask = img > 0
        img[~mask] = 1.0
        img = img.sub_(img.min()).div_(img.max()) * 200.0 / 255.0
        img[~mask] = 1.0
        return self._normalize(img.repeat(3, 1, 1))
