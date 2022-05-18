from mmcv import Config
from mmdet3d.models import build_model

from torch import nn

import clip.model

from lidar_clippin.sst_encoder_only import model as sst_model_conf


def build_sst(config_path):
    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    return model


class LidarEncoder(nn.Module):
    def __init__(self, sst_config_path):
        super().__init__()
        self._sst = build_sst(sst_config_path)
        self._pooler = clip.model.AttentionPool2d(
            spacial_dim=sst_model_conf["backbone"]["output_shape"][0],
            embed_dim=sst_model_conf["backbone"]["conv_out_channel"],
            num_heads=8,
        )

    def forward(self, point_cloud):
        lidar_features = self._sst.extract_feat(point_cloud, None)
        pooled_feature = self._pooler(lidar_features)
        return pooled_feature


if __name__ == "__main__":
    model = LidarEncoder("sst_encoder_only.py")
    import torch

    model.to("cuda")
    points = [torch.rand(100, 3).cuda() for _ in range(16)]
    out = model(points)
    print(out[0].shape)
