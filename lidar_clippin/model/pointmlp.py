from pointMLP.classification_ModelNet40.models.pointmlp import Model as PointMlpModel

import torch.nn as nn
import torch.nn.functional as F


def _max_pool(x):
    return F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)


class LidarEncoderPointMLP(PointMlpModel):
    def __init__(self, points=8192, out_dim=512) -> None:
        super().__init__(
            points=points,
            class_num=out_dim,
            embed_dim=32,
            groups=1,
            res_expansion=0.25,
            activation="relu",
            bias=False,
            use_xyz=False,
            normalize="anchor",
            dim_expansion=[2, 2, 2, 1],
            pre_blocks=[1, 1, 2, 1],
            pos_blocks=[1, 1, 2, 1],
            k_neighbors=[24, 24, 24, 24],
            reducers=[4, 4, 4, 2],
        )
        last_channel = self.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, out_dim),
            nn.BatchNorm1d(out_dim),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(out_dim, out_dim),
        )
        # self.pool = AttentionPool1d()
        self.pool = _max_pool

    def forward(self, point_clouds):
        x = self._subsample_points(point_clouds)
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embedding(x)  # B,D,N
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]

        x = self.pool(x)
        x = self.classifier(x)
        return x

    def _subsample_points(point_clouds):
        torch.tensor([pc[torch.randperm(pc.shape[0])[:8192]] for pc in point_clouds])


if __name__ == "__main__":
    model = LidarEncoderPointMLP()
    import torch

    model.to("cuda")
    points = [torch.rand(100, 4).cuda() for _ in range(16)]
    out = model(points)
    print(out.shape)
