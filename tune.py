"""Module for Hyperparameter tuning"""
import gc

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger

import wandb
from src.data.lightning import FGRPretrainDataModule
from src.model.lightning import FGRPretrainLightning

sweep_config = {
    "method": "bayes",  # Bayesian search
    "metric": {"name": "val_loss", "goal": "minimize"},  # We want to minimize val_loss
    "parameters": {
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-2,
        },
        "max_lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-1,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-3,
            "max": 0.9,
        },
        "dropout": {"values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
        "n_layers": {"values": [1, 2, 3, 4, 5]},
        "hidden_size": {"values": [128, 256, 512, 1024, 2048]},
    },
}


def run_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config

        gc.collect()
        torch.cuda.empty_cache()

        seed_everything(seed=123)

        fg_input_dim = 2786
        mfg_input_dim = 3000
        method = "FGR"
        dropout = wandb.config.dropout
        weight_decay = wandb.config.weight_decay
        lr = wandb.config.lr
        max_lr = wandb.config.max_lr
        n_layers = wandb.config.n_layers
        hidden_size = wandb.config.hidden_size
        hidden_dims = []
        for _ in range(n_layers):
            hidden_dims.append(hidden_size)
            hidden_size = hidden_size // 2

        datamodule = FGRPretrainDataModule("datasets/processed/", "chembl", 32, 8, True, method)
        model = FGRPretrainLightning(
            fg_input_dim,
            mfg_input_dim,
            method,
            hidden_dims,
            dropout,
            lr,
            weight_decay,
            max_lr,
        )
        wandb_logger = WandbLogger()

        trainer = Trainer(
            logger=wandb_logger,
            accelerator="gpu",
            devices=-1,
            max_epochs=3,
            precision=16,
        )
        trainer.fit(model, datamodule)
        del model
        del wandb_logger
        del datamodule
        del trainer

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="Pretrain_sweep")
    wandb.agent(sweep_id, function=run_sweep, count=100)
