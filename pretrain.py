"""Module for pretraining"""
from src.data.lightning import FGRPretrainDataModule
from src.model.lightning import FGRPretrainLightning
from pytorch_lightning.cli import LightningCLI


class PretrainLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.method", "data.method")


if __name__ == "__main__":
    cli = PretrainLightningCLI(
        model_class=FGRPretrainLightning,
        datamodule_class=FGRPretrainDataModule,
        save_config_callback=None,
    )
