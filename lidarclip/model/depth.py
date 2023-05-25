import clip
from einops import rearrange
import torch.nn as nn
import torch
from typing import List

from lidarclip.loader import build_loader
from lightly.loss.ntx_ent_loss import NTXentLoss

try:
    from lidarclip.depth_renderer import DepthRenderer
except ImportError:
    print("DepthRenderer not available, depth rendering will not work")


class DepthLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.criterion = NTXentLoss(temperature=temperature)
        self.weights = nn.Parameter(torch.ones([]))

    def forward(self, img_feats, lidar_feats):
        """Compute the CLIP2Point loss.

        Args:
            img_feats (torch.Tensor): Image features from CLIP
            lidar_feats (torch.Tensor): Lidar features from PointNet. Can be either
                a single depth map (B,C) or a pair of depth maps (B,2,C).

        Returns:
            torch.Tensor: Loss value
        """
        if img_feats.shape[0] == 1:
            print("Warning: Batch size of 1, contrastive loss will be 0")
        if lidar_feats.shape[1] == 1:
            # this is the simple settings without multiple depth views
            return self.criterion(lidar_feats, img_feats)
        assert lidar_feats.shape[1] == 2, "Mult-view depth map must have exactly 2 views"
        depth1_feats = lidar_feats[:, 0]
        depth2_feats = lidar_feats[:, 1]
        depth_feats = (depth1_feats + depth2_feats) * 0.5
        img_loss = self.criterion(depth_feats, img_feats)
        # print("img ", img_loss)
        depth_loss = self.criterion(depth1_feats, depth2_feats)
        # print("depth: ", depth_loss)
        weighted_loss = img_loss + depth_loss / (self.weights**2) + torch.log(self.weights + 1)
        # print("weighted: ", weighted_loss)
        return weighted_loss


class DepthEncoder(nn.Module):
    def __init__(self, clip_model_name, depth_aug=False):
        super().__init__()
        self._clip, _ = clip.load(clip_model_name)
        self._depth_renderer = DepthRenderer(aug=depth_aug)

    def forward(self, point_cloud, no_pooling=False, return_attention=False, metadata=dict()):
        cam_intrinsic = metadata["cam_intrinsic"].to(point_cloud[0].device)
        img_size = metadata["img_size"]
        point_cloud = self.render_depth(
            point_cloud, cam_intrinsic, img_size, aug=self._depth_renderer.aug
        )
        assert (
            len(point_cloud.shape) == 5
        ), "Point cloud must have shape (B, V, C, H, W), where V is the number of views"
        n_views = point_cloud.shape[1]
        point_cloud = rearrange(point_cloud, "b v c h w -> (b v) c h w")
        assert point_cloud.shape[1:] == (
            3,
            224,
            224,
        ), "Point cloud must rendered into a depth map of shape (..., 3, 224, 224)"
        feats = self._clip.encode_image(point_cloud)
        return rearrange(feats, "(b v) c -> b v c", v=n_views), None  # no attention for now

    def render_depth(
        self,
        point_clouds: List[torch.Tensor],
        cam_intrinsics: torch.Tensor,
        img_size: torch.Tensor,
        aug: bool = False,
    ) -> torch.Tensor:
        max_num_points = max([point_cloud.shape[0] for point_cloud in point_clouds])
        # assert all img_size are the same
        assert all(img_size[:, 0] == img_size[0, 0]) and all(
            img_size[:, 1] == img_size[0, 1]
        ), "All images must have the same size"
        img_size = tuple(img_size[0].cpu().to(torch.float).tolist())
        # pad point clouds to the same size
        for i, point_cloud in enumerate(point_clouds):
            num_points = point_cloud.shape[0]
            point_clouds[i] = nn.functional.pad(point_cloud, (0, 0, 0, max_num_points - num_points))
            point_clouds[i][num_points:, 0] = -10  # put padded points behind the camera
        point_clouds = torch.stack(point_clouds)
        pc_shape = point_clouds.shape  # (B, N_points, 3)
        if aug:
            cam_intrinsics_shape = cam_intrinsics.shape
            # repeat point clouds and cam_intrinsics twice
            point_clouds = (
                point_clouds.unsqueeze(1)
                .repeat(1, 2, 1, 1)
                .view(pc_shape[0] * 2, pc_shape[1], pc_shape[2])
            )
            cam_intrinsics = (
                cam_intrinsics.unsqueeze(1)
                .repeat(1, 2, 1, 1)
                .view(cam_intrinsics_shape[0] * 2, cam_intrinsics_shape[1], cam_intrinsics_shape[2])
            )

        rendered_pc = self._depth_renderer(point_clouds, cam_intrinsics, img_size)
        if aug:
            rendered_pc = rearrange(rendered_pc, "(b v) c h w -> b v c h w", v=2)
        else:
            # add view dimension
            rendered_pc = rendered_pc.unsqueeze(1)
        return rendered_pc


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    _, clip_preprocess = clip.load("ViT-B/32")
    means = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cpu")
    stds = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cpu")
    depth_encoder = DepthEncoder("ViT-B/32")

    nuscenes_datadir = "/Users/s0000960/data/nuscenes"
    once_datadir = "/Users/s0000960/data/once"
    loader = build_loader(
        clip_preprocess=clip_preprocess,
        num_workers=0,
        batch_size=2,
        dataset_name="once",
        once_split="val",
        once_datadir=once_datadir,
        nuscenes_datadir=nuscenes_datadir,
        nuscenes_split="mini",
        shuffle=True,
    )
    iter_loader = iter(loader)

    for i, (images, lidars) in enumerate(iter_loader):
        depths = depth_encoder(lidars)
        image = rearrange(images[0], "c h w -> h w c") * stds + means
        depth = rearrange(depths[0], "c h w -> h w c") * stds + means
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(depth)
        plt.axis("equal")
        plt.show()
