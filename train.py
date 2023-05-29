import argparse
import os

import clip
import pytorch_lightning as pl
import torch
from clip.model import CLIP
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F

try:
    from torch_optimizer import Lamb
except ImportError:
    print("Got ImportError for torch_optmizer, Lamb not available")

from lidarclip.loader import build_loader

try:
    from lidarclip.model.sst import SECOND, LidarEncoderSST
except ImportError:
    print("Got ImportError for SST model, not importing")
try:
    from lidarclip.model.depth import DepthEncoder, DepthLoss
except ImportError:
    print("Got ImportError for Depth model and loss, not importing")


class LidarClip(pl.LightningModule):
    def __init__(
        self,
        lidar_encoder: torch.nn.Module,
        clip_model: CLIP,
        batch_size: int,
        epoch_size: int,
        loss: str = "mse",
        optimizer: str = "adam",
        peak_lr: float = 1e-3,
        do_gather: bool = False,
    ):
        super().__init__()
        self.lidar_encoder = lidar_encoder
        self.clip = clip_model
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        for param in self.clip.parameters():
            param.requires_grad = False
        if loss == "mse":
            self.loss_func = F.mse_loss
        elif loss == "cosine":
            self.loss_func = lambda x, y: -F.cosine_similarity(x, y).mean()
        elif loss == "depth":
            self.loss_func = DepthLoss()
        else:
            raise ValueError(f"Loss {loss} not supported")
        self._optimizer_type = optimizer
        self._peak_lr = peak_lr
        self.do_gather = do_gather

    def training_step(self, batch, batch_idx):
        image, point_cloud, metadata = batch
        with torch.no_grad():
            # This could in principle be pre-computed, but that would
            # break any joint image-lidar augmentations
            image_features = self.clip.encode_image(image)
        lidar_features, _ = self.lidar_encoder(point_cloud, metadata=metadata)
        if self.do_gather:
            image_features_shape = image_features.shape
            lidar_features_shape = lidar_features.shape
            image_features = self.all_gather(image_features, sync_grads=False).view(
                -1, *image_features_shape[1:]
            )
            lidar_features = self.all_gather(lidar_features, sync_grads=True).view(
                -1, *lidar_features_shape[1:]
            )
        if len(lidar_features.shape) == 3 and not isinstance(self.loss_func, DepthLoss):
            assert lidar_features.shape[1] == 1, "Lidar features should only have a single view"
            lidar_features = lidar_features.squeeze(1)
        loss = self.loss_func(image_features, lidar_features)
        self.log("train_loss", loss.detach())
        return loss

    def configure_optimizers(self):
        # Epoch_size is number of batches/steps per epoch
        if type(self.trainer.limit_train_batches) == float:
            epoch_size = int(self.epoch_size * self.trainer.limit_train_batches)
        elif type(self.trainer.limit_train_batches) == int:
            epoch_size = self.trainer.limit_train_batches
        elif self.trainer.limit_train_batches is None:
            epoch_size = int(self.epoch_size)
        steps_per_epoch = epoch_size // self.trainer.accumulate_grad_batches

        if self._optimizer_type == "lamb":
            optimizer = Lamb(self.lidar_encoder.parameters(), lr=0.006, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(0.1 * steps_per_epoch),
                T_mult=1,
                eta_min=max(1e-2 * 1e-3, 1e-6),
                last_epoch=-1,
            )
        elif self._optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.lidar_encoder.parameters(), lr=1e-5)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self._peak_lr,
                # total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                steps_per_epoch=steps_per_epoch,
                epochs=self.trainer.max_epochs,
            )
        else:
            raise ValueError(f"Optimizer {self._optimizer_type} not supported")
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]


def train(
    name,
    checkpoint_save_dir,
    checkpoint_path,
    batch_size,
    num_workers,
    epochs,
    load_only_model=False,
    resume_wandb_logging=False,
    clip_model_name="ViT-B/32",
    loss_function="mse",
    once_datadir="/once",
    once_split="train",
    nuscenes_datadir="/nuscenes",
    nuscenes_split="train",
    dataset_name="once",
    model="sst",
):
    """Train the model."""
    clip_model, clip_preprocess = clip.load(clip_model_name, jit=False)
    clip_model.eval()
    clip_embed_dim = clip_model.visual.output_dim
    if model == "sst":
        lidar_encoder = LidarEncoderSST(
            "lidarclip/model/sst_encoder_only_config.py", clip_embed_dim
        )
    elif model == "second":
        lidar_encoder = SECOND("lidarclip/model/centerpoint_encoder_only.py", clip_embed_dim)
    elif model == "depth":
        lidar_encoder = DepthEncoder(clip_model_name, depth_aug=(loss_function == "depth"))
    else:
        raise NotImplementedError(f"model {model} not implemented yet")
    available_gpus = torch.cuda.device_count() or None
    accelerator = "gpu" if available_gpus else "cpu"
    devices = available_gpus if available_gpus else 1

    train_loader = build_loader(
        clip_preprocess=clip_preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        once_datadir=once_datadir,
        once_split=once_split,
        nuscenes_datadir=nuscenes_datadir,
        nuscenes_split=nuscenes_split,
        dataset_name=dataset_name,
        depth_rendering="aug" if model == "depth" else False,
    )

    wandb_id = None
    wand_resume = False
    model = LidarClip(
        lidar_encoder,
        clip_model,
        batch_size,
        len(train_loader) / devices,
        loss_function,
        args.optimizer,
        args.peak_lr,
        args.do_gather,
    )
    if len(checkpoint_path) and resume_wandb_logging:
        wandb_id = checkpoint_path.split("/")[-2]
        wand_resume = "must"

    if len(checkpoint_path) and load_only_model:
        model = LidarClip.load_from_checkpoint(
            checkpoint_path,
            lidar_encoder=lidar_encoder,
            clip_model=clip_model,
            batch_size=batch_size,
            epoch_size=len(train_loader) / devices,
            loss=loss_function,
        )
        checkpoint_path = None

    elif len(checkpoint_path) == 0:
        checkpoint_path = None

    wandb_logger = WandbLogger(
        project="lidarclip",
        entity="agp",
        name=name,
        resume=wand_resume,
        id=wandb_id,
        allow_val_change=True,
    )

    if checkpoint_save_dir:
        checkpoint_save_dir = os.path.join(checkpoint_save_dir, str(wandb_logger.version))
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_save_dir,
        save_top_k=3,
        monitor="train_loss",
        save_last=True,
        every_n_train_steps=250,
        save_on_train_epoch_end=True,
    )
    learningrate_callback = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        precision=16,
        accelerator=accelerator,
        devices=devices,
        limit_train_batches=None,
        max_epochs=epochs,
        logger=wandb_logger,
        strategy="ddp",
        callbacks=[checkpoint_callback, learningrate_callback],
        resume_from_checkpoint=checkpoint_path,
    )
    if trainer.global_rank == 0:
        old_id = wandb_logger.experiment.config.get("slurm-id", "")
        curr_id = os.environ.get("SLURM_JOB_ID", "unknown")
        new_id = old_id + "-" + curr_id if len(old_id) else curr_id
        wandb_logger.experiment.config.update({"slurm-id": new_id}, allow_val_change=True)

    trainer.fit(model=model, train_dataloaders=train_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint-save-dir", default=None)
    parser.add_argument("--checkpoint", required=False, default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--load-only-model", action="store_true")
    parser.add_argument("--resume-wandb-logging", action="store_true")
    parser.add_argument("--clip-model", default="ViT-L/14", help="which clip model to use")
    parser.add_argument(
        "--loss-function",
        default="mse",
        help="which loss function to use",
        choices=("cosine", "mse", "depth"),
    )
    parser.add_argument("--once-datadir", default="/once")
    parser.add_argument("--once-split", default="train")
    parser.add_argument("--nuscenes-datadir", default="/nuscenes")
    parser.add_argument("--nuscenes-split", default="train")
    parser.add_argument("--dataset-name", default="once")
    parser.add_argument("--model", default="sst")
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--peak-lr", default=1e-3, type=float)
    parser.add_argument("--do-gather", action="store_true")
    args = parser.parse_args()
    assert args.name, "Empty name is not allowed"
    return args


if __name__ == "__main__":
    args = parse_args()
    train(
        args.name,
        args.checkpoint_save_dir,
        args.checkpoint,
        args.batch_size,
        args.workers,
        args.epochs,
        args.load_only_model,
        args.resume_wandb_logging,
        clip_model_name=args.clip_model,
        loss_function=args.loss_function,
        once_datadir=args.once_datadir,
        once_split=args.once_split,
        nuscenes_datadir=args.nuscenes_datadir,
        nuscenes_split=args.nuscenes_split,
        dataset_name=args.dataset_name,
        model=args.model,
    )
