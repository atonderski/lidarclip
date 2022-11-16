from mmcv import Config
from mmdet3d.models import build_model

import torch
from torch import nn

from lidarclip.model.attention_pool import AttentionPool2d
from lidarclip.model.sst_encoder_only_config import model as sst_model_conf


def build_sst(config_path):
    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    return model


class LidarEncoderSST(nn.Module):
    def __init__(self, sst_config_path, clip_embedding_dim=512):
        super().__init__()
        self._sst = build_sst(sst_config_path)
        self._pooler = AttentionPool2d(
            spacial_dim=sst_model_conf["backbone"]["output_shape"][0],
            embed_dim=clip_embedding_dim,
            num_heads=8,
            input_dim=sst_model_conf["backbone"]["conv_out_channel"],
        )

    def forward(self, point_cloud, no_pooling=False, return_attention=False):
        lidar_features = self._sst.extract_feat(point_cloud, None)[0]  # bs, d, h, w
        pooled_feature, attn_weights = self._pooler(lidar_features, no_pooling, return_attention)
        return pooled_feature, attn_weights


if __name__ == "__main__":
    model = LidarEncoderSST("sst_encoder_only.py")
    import torch

    model.to("cuda")
    points = [torch.rand(100, 4).cuda() for _ in range(16)]
    out = model(points)
    print(out.shape)
