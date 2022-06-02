import argparse

import pytorch_lightning as pl
from mmcv.runner import load_checkpoint
from pytorch_lightning.loggers import WandbLogger

import torch
from torch.nn import functional as F

import clip
from clip.model import CLIP

from lidar_clippin.loader import build_loader
from lidar_clippin.model import LidarEncoder


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


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
        loss = F.mse_loss(l2norm(image_features), l2norm(lidar_features))
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.lidar_encoder.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
        )
        return [optimizer], [scheduler]


def train(data_dir, name, checkpoint):
    """Train the model."""
    clip_model, clip_preprocess = clip.load("ViT-B/32")
    lidar_encoder = LidarEncoder("lidar_clippin/sst_encoder_only.py")
    model = LidarClippin(lidar_encoder, clip_model)
    if len(checkpoint):
        load_checkpoint(model, checkpoint, map_location="cpu")
    available_gpus = torch.cuda.device_count() or None
    num_workers = available_gpus * 4 if available_gpus else 8
    train_loader = build_loader(data_dir, clip_preprocess, batch_size=32, num_workers=num_workers)

    wandb_logger = WandbLogger(project="lidar-clippin", entity="agp", name=name)
    accelerator = "gpu" if available_gpus else "cpu"
    devices = available_gpus if available_gpus else 1
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        limit_train_batches=1.0,
        max_epochs=5,
        logger=wandb_logger,
        strategy="ddp",
    )
    trainer.fit(model=model, train_dataloaders=train_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=False, default="")
    args = parser.parse_args()
    assert args.name  # empty name is not allowed
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.name, args.checkpoint)
