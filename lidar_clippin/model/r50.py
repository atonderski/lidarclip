import torch
from torch import nn

from clip.model import ModifiedResNet

from lidar_clippin.model.attention_pool import AttentionPool2d


class LidarEncoderBase(nn.Module):
    def __init__(self, image_resolution: int, channels: int):
        super().__init__()
        width = 64
        heads = width * 32 // 64
        self.resnet = ModifiedResNet(
            layers=(3, 4, 6, 3),  # TODO
            output_dim=512,
            heads=heads,
            input_resolution=image_resolution,
            width=width,
        )
        self.resnet.conv1 = nn.Conv2d(
            channels, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.resnet.attnpool = AttentionPool2d(
            image_resolution // 32, input_dim=2048, embed_dim=2048, num_heads=heads, output_dim=512
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        std = self.resnet.attnpool.c_proj.in_features**-0.5
        nn.init.normal_(self.resnet.attnpool.q_proj.weight, std=std)
        nn.init.normal_(self.resnet.attnpool.k_proj.weight, std=std)
        nn.init.normal_(self.resnet.attnpool.v_proj.weight, std=std)
        nn.init.normal_(self.resnet.attnpool.c_proj.weight, std=std)

        for resnet_block in [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
        ]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def forward(self, bev_features):
        return self.resnet(bev_features)


if __name__ == "__main__":
    # import clip
    # clip.load("RN50")
    input_res, channels = 256, 30
    model = LidarEncoderBase(input_res, channels)
    # points = [torch.rand(100, 4).cuda() for _ in range(16)]
    bev_features = torch.rand(2, channels, input_res, input_res)
    out = model(bev_features)
    print(out.shape)
