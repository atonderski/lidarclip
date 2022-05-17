import clip
import torch
from torch.nn import functional as F

from model import LidarEncoder


def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    lidar_encoder = LidarEncoder()
    dataloader = build_loader(clip_preprocess)

    optimizer = torch.optim.AdamW(lidar_encoder.parameters())

    for image, point_cloud in dataloader:
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
        optimizer.zero_grad()
        lidar_features = lidar_encoder(point_cloud)
        loss = F.l1_loss(image_features, lidar_features)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    train()
