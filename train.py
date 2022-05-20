import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import clip
from clip.model import CLIP

from lidar_clippin.loader import OnceImageLidarDataset
from lidar_clippin.model import LidarEncoder


class LidarClippin(pl.LightningModule):
    def __init__(self, lidar_encoder: LidarEncoder, clip_model: CLIP):
        super().__init__()
        self.lidar_encoder = lidar_encoder
        self.clip = clip_model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        image, point_cloud = batch
        with torch.no_grad():
            image_features = self.clip.encode_image(image)
        lidar_features = self.lidar_encoder(point_cloud)
        loss = F.l1_loss(image_features, lidar_features)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.lidar_encoder.parameters(), lr=3e-4)
        return optimizer


def train(data_dir):
    """Train the model."""
    clip_model, clip_preprocess = clip.load("ViT-B/32")
    lidar_encoder = LidarEncoder("lidar_clippin/sst_encoder_only.py")
    model = LidarClippin(lidar_encoder, clip_model)

    dataset = OnceImageLidarDataset(data_dir, clip_preprocess)
    train_loader = DataLoader(dataset)

    wandb_logger = WandbLogger(project="lidar-clippin")
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir)
