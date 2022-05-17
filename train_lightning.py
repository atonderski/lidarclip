from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import clip
from clip.model import CLIP

from loader import OnceImageLidarDataset
from model import LidarEncoder


class LidarClippin(pl.LightningModule):
    def __init__(self, lidar_encoder: LidarEncoder, clip_model: CLIP):
        super().__init__()
        self.lidar_encoder = lidar_encoder
        self.clip = clip_model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        image, point_cloud = batch
        image_features = self.clip.encode_image(image)
        lidar_features = self.lidar_encoder(point_cloud)
        loss = F.l1_loss(image_features, lidar_features)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.lidar_encoder.parameters(), lr=3e-4)
        return optimizer


def train():
    clip_model, clip_preprocess = clip.load("ViT-B/32")
    lidar_encoder = LidarEncoder()
    model = LidarClippin(lidar_encoder, clip_model)
    dataset = OnceImageLidarDataset("/Users/s0000960/data/once")
    train_loader = DataLoader(dataset)
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == '__main__':
    train()
