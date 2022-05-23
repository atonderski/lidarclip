from mmcv import Config
from mmdet3d.models import build_model

import torch
import torch.nn.functional as F
from torch import nn

import clip.model

from lidar_clippin.sst_encoder_only import model as sst_model_conf


class AttentionPool2d(nn.Module):
    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads: int,
        input_dim: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        key_padding_mask = (x.abs().sum(dim=-1) == 0).permute(1, 0)  # (HW)NC -> (HW)N -> N(HW)
        x = self.in_proj(x)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[0:1],
            key=x[1:],
            value=x[1:],
            key_padding_mask=key_padding_mask,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


def build_sst(config_path):
    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    return model


class LidarEncoder(nn.Module):
    def __init__(self, sst_config_path):
        super().__init__()
        self._sst = build_sst(sst_config_path)
        self._pooler = AttentionPool2d(
            spacial_dim=sst_model_conf["backbone"]["output_shape"][0],
            embed_dim=512,
            num_heads=8,
            input_dim=sst_model_conf["backbone"]["conv_out_channel"],
        )
        # self._pooler = lambda x: x.mean(dim=(-1, -2))
        # self._pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, point_cloud):
        lidar_features = self._sst.extract_feat(point_cloud, None)[0]  # bs, d, h, w
        pooled_feature = self._pooler(lidar_features)
        # pooled_feature = lidar_features.mean(dim=(-1, -2))
        # lidar_features = lidar_features.flatten(2) #bs, d, h*w
        # pooled_feature = self._pooler(lidar_features) #bs, d, 1
        # pooled_feature = pooled_feature.flatten(1) #bs, d
        return pooled_feature


if __name__ == "__main__":
    model = LidarEncoder("sst_encoder_only.py")
    import torch

    model.to("cuda")
    points = [torch.rand(100, 4).cuda() for _ in range(16)]
    out = model(points)
    print(out.shape)
