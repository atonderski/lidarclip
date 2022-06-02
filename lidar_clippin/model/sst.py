from mmcv import Config
from mmdet3d.models import build_model

import torch
import torch.nn.functional as F
from torch import nn

from lidar_clippin.model.attention_pool import AttentionPool2d
from lidar_clippin.sst_encoder_only_config import model as sst_model_conf


def build_sst(config_path):
    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    return model


class LidarEncoderSST(nn.Module):
    def __init__(self, sst_config_path):
        super().__init__()
        self._sst = build_sst(sst_config_path)
        self._pooler = AttentionPool2d(
            spacial_dim=sst_model_conf["backbone"]["output_shape"][0],
            embed_dim=512,
            num_heads=8,
            input_dim=sst_model_conf["backbone"]["conv_out_channel"],
        )
        # self._pooler = _mean_reduce
        # self._pooler = nn.AdaptiveAvgPool1d(1)
        # self._linear = nn.Linear(128, 512)

    def forward(self, point_cloud):
        lidar_features = self._sst.extract_feat(point_cloud, None)[0]  # bs, d, h, w
        pooled_feature = self._pooler(lidar_features)
        # pooled_feature = lidar_features.mean(dim=(-1, -2))
        # lidar_features = lidar_features.flatten(2) #bs, d, h*w
        # pooled_feature = self._pooler(lidar_features) #bs, d, 1
        # pooled_feature = pooled_feature.flatten(1) #bs, d
        # pooled_feature = self._linear(pooled_feature)
        return pooled_feature


if __name__ == "__main__":
    model = LidarEncoderSST("sst_encoder_only.py")
    import torch

    model.to("cuda")
    points = [torch.rand(100, 4).cuda() for _ in range(16)]
    out = model(points)
    print(out.shape)
