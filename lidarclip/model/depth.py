import clip
from einops import rearrange
import torch.nn as nn
import torch

from lidarclip.loader import build_loader
from lightly.loss.ntx_ent_loss import NTXentLoss


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
        print("img ", img_loss)
        depth_loss = self.criterion(depth1_feats, depth2_feats)
        print("depth: ", depth_loss)
        weighted_loss = img_loss + depth_loss / (self.weights**2) + torch.log(self.weights + 1)
        print("weighted: ", weighted_loss)
        return weighted_loss


class DepthEncoder(nn.Module):
    def __init__(self, clip_model_name):
        super().__init__()
        self._clip, _ = clip.load(clip_model_name)

    def forward(self, point_cloud, no_pooling=False, return_attention=False):
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

    # def render_depth(self, point_clouds: List[torch.Tensor], aug: bool = False) -> torch.Tensor:
    #     depth1s = []
    #     depth2s = []
    #     for point_cloud in point_clouds:
    #         # from kitti forward, left, up to torch3d left, up, forward  (reflectance is not used)
    #         point_cloud = point_cloud[:, [1, 2, 0]]
    #     res = self._clip.visual.input_resolution


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
