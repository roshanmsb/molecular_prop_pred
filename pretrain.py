"""Module for pretraining"""
from lightning.pytorch.cli import LightningCLI
import torch

from src.data.data_module import FGRPretrainDataModule
from src.model.lightning_module import FGRPretrainLightning


class PretrainLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.method", "data.method")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    cli = PretrainLightningCLI(
        model_class=FGRPretrainLightning,
        datamodule_class=FGRPretrainDataModule,
        save_config_callback=None,
    )
