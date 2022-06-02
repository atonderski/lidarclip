import torch
import torch.nn as nn
import torch.nn.functional as F

from pointMLP.classification_ModelNet40.models.pointmlp import ConvBNReLU1D, Model


def _max_pool(x):
    return F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)


class LidarEncoderPointMLP(Model):
    def __init__(self, points=8192, out_dim=512, reducers=(4, 4, 4, 2)) -> None:
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
            reducers=reducers,
        )
        # Change embedding to make use of reflectance as well
        self.embedding = ConvBNReLU1D(4, 32, bias=False, activation="relu")
        # Change classifier to avoid downprojecting
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
        with torch.no_grad():
            x = self._resample_points(point_clouds)  # [b,n,4]
            xyz = x[..., :3]  # [b,n,3]
            x = x.permute(0, 2, 1)  # [b,3,n]
        x = self.embedding(x)  # [b,d,n]
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]
        x = self.pool(x)  # [b,d]
        x = self.classifier(x)  # [b,d]
        return x

    def _resample_points(self, point_clouds):
        resampled = []
        for pc in point_clouds:
            if pc.shape[0] < self.points:
                num_missing = self.points - pc.shape[0]
                print(f"Adding {num_missing} points")
                pc = torch.cat((pc, pc[torch.randint(0, pc.shape[0], (num_missing,))]))
                print(pc.shape)
            else:
                resampled.append(pc[torch.randperm(pc.shape[0])[: self.points]])
        return torch.stack(resampled)


if __name__ == "__main__":
    model = LidarEncoderPointMLP()
    import torch

    model.to("cuda")
    points = [torch.rand(100, 4).cuda() for _ in range(16)]
    out = model(points)
    print(out.shape)
