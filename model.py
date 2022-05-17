from torch import nn
from mmcv import Config
from mmdet3d.models import build_model


def build_sst(config_path):
    config_path = "SST/configs/sst_refactor/sst_encoder_only.py"
    cfg = Config.fromfile(config_path)
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    return model

class LidarEncoder(nn.Module):

    def __init__(self, sst_config_path):
        super().__init__()
        self._sst = build_sst(sst_config_path)
        self._pooler = build_pooler()

    def forward(self, point_cloud):
        lidar_features = self._sst(point_cloud)
        pooled_feature = self._pooler(lidar_features)
        return pooled_feature

