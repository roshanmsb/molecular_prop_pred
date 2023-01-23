"""Module for Hyperparameter tuning"""
import gc

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger

import wandb
from src.data.data_module import FGRPretrainDataModule
from src.model.lightning_module import FGRPretrainLightning


def run_sweep(config=None):
    with wandb.init(config=config):  # type: ignore
        config = wandb.config

        gc.collect()
        torch.cuda.empty_cache()

        seed_everything(seed=123)

        fg_input_dim = 2786
        mfg_input_dim = 3000
        method = "FGR"
        weight_decay = wandb.config.weight_decay
        lr = wandb.config.lr
        max_lr = wandb.config.max_lr
        n_layers = wandb.config.n_layers
        hidden_size = wandb.config.hidden_size
        bottleneck_dim = wandb.config.bottleneck_dim
        hidden_dims = []
        for _ in range(n_layers):
            hidden_dims.append(hidden_size)
            hidden_size = hidden_size // 2

        datamodule = FGRPretrainDataModule("datasets/processed/", "chembl", 256, 32, True, method)
        model = FGRPretrainLightning(
            fg_input_dim,
            mfg_input_dim,
            method,
            hidden_dims,
            bottleneck_dim,
            lr,
            weight_decay,
            max_lr,
        )
        wandb_logger = WandbLogger()

        trainer = Trainer(
            logger=wandb_logger, accelerator="gpu", devices=1, max_epochs=2, precision=16,
        )
        trainer.fit(model, datamodule)
        del model
        del wandb_logger
        del datamodule
        del trainer

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_sweep()
