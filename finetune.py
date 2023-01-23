from lightning.pytorch.cli import LightningCLI

from src.data.data_module import FGRDataModule
from src.model.lightning_module import FGRFinetuneLightning


class FinetuneLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.method", "data.method")


if __name__ == "__main__":
    cli = FinetuneLightningCLI(
        model_class=FGRFinetuneLightning,
        datamodule_class=FGRDataModule,
        save_config_callback=None,
        run=False,
    )
    cli.trainer.fit(cli.model, cli.datamodule)  # type: ignore
    cli.trainer.test(cli.model, cli.datamodule, ckpt_path="best")
