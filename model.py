from torch import nn


class LidarEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self._sst = build_sst()
        self._pooler = build_pooler()

    def forward(self, point_cloud):
        lidar_features = self._sst(point_cloud)
        pooled_feature = self._pooler(lidar_features)
        return pooled_feature

