"""Module for pretraining"""
from src.data.datamodules import FGRPretrainDataModule
from src.model.lightning_modules import FGRPretrainLightning
from pytorch_lightning.cli import LightningCLI


if __name__ == "__main__":
    cli = LightningCLI(
        model_class=FGRPretrainLightning,
        datamodule_class=FGRPretrainDataModule,
        save_config_callback=None,
    )
