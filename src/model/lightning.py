from typing import List
import torch
from torch import nn
from pytorch_lightning import LightningModule
import torchmetrics
from src.model.network import FGRModel, FGRPretrainModel


class FGRLightning(LightningModule):
    """Lightning module for training"""

    def __init__(
        self,
        fg_input_dim: int,
        mfg_input_dim: int,
        num_input_dim: int,
        num_tasks: int,
        method: str,
        regression: bool,
        hidden_dims: List[int],
        bottleneck_dim: int,
        output_dims: List[int],
        dropout: float,
        lr: float,
        weight_decay: float,
        max_lr: float,
        **kwargs
    ):
        """Initialize the FGR model

        Args:
            fg_input_dim (int): Input dimension for FG
            mfg_input_dim (int): Input dimension for MFG
            num_input_dim (int): Input dimension for RDKit features
            num_tasks (int): Number of tasks for each dataset
            method (str): Representation method to train
            regression (bool):Whether the task is regression or classification
            hidden_dims (List[int]): Dimensions for each layer
            bottleneck_dim (int): Dimension of bottleneck layer
            output_dims (List[int]): Dimensions for each layer in predictor
            dropout (float): Dropout for each layer
            lr (float): Learning rate for optimizer
            weight_decay (float): Weight decay for optimizer
            max_lr (float): Maximum learning rate for scheduler
        """

        super().__init__()
        self.save_hyperparameters()
        self.net = FGRModel(
            fg_input_dim,
            mfg_input_dim,
            num_input_dim,
            hidden_dims,
            bottleneck_dim,
            output_dims,
            num_tasks,
            dropout,
            method,
        )
        self.l_r = lr
        self.method = method
        self.regression = regression
        self.weight_decay = weight_decay
        self.max_lr = max_lr
        if self.regression:
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.recon_loss = nn.BCEWithLogitsLoss()

        if self.regression:
            self.train_mae = torchmetrics.MeanAbsoluteError()
            self.val_mae = torchmetrics.MeanAbsoluteError()
            self.test_mae = torchmetrics.MeanAbsoluteError()
            self.train_mse = torchmetrics.MeanSquaredError()
            self.val_mse = torchmetrics.MeanSquaredError()
            self.test_mse = torchmetrics.MeanSquaredError()
            self.train_r2 = torchmetrics.R2Score(num_outputs=num_tasks)
            self.val_r2 = torchmetrics.R2Score(num_outputs=num_tasks)
            self.test_r2 = torchmetrics.R2Score(num_outputs=num_tasks)
        else:
            self.train_auc = torchmetrics.AUROC(num_classes=num_tasks)
            self.val_auc = torchmetrics.AUROC(num_classes=num_tasks)
            self.test_auc = torchmetrics.AUROC(num_classes=num_tasks)
            self.train_f1 = torchmetrics.F1Score(num_classes=num_tasks)
            self.val_f1 = torchmetrics.F1Score(num_classes=num_tasks)
            self.test_f1 = torchmetrics.F1Score(num_classes=num_tasks)

    def forward(self, fgr=None, num_feat=None):
        if self.method != "FGR_desc":
            preds = self.net(fgr)
        else:
            preds = self.net(fgr, num_feat=num_feat)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.l_r, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.max_lr, total_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        if self.method != "FGR_desc":
            fgr, y_true = batch
            y_pred, recon = self(fgr)
        else:
            fgr, num_features, y_true = batch
            y_pred, recon = self(fgr, num_features)

        loss_r_pre = self.recon_loss(recon, fgr)
        if self.regression:
            y_true = y_true.float()
        criterion_loss = self.criterion(y_pred, y_true)
        loss = criterion_loss + loss_r_pre
        self.log(
            "train_loss", criterion_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        if self.regression:
            self.train_mse.update(y_pred, y_true)
            self.train_mae.update(y_pred, y_true)
            self.train_r2.update(y_pred, y_true)
            self.log(
                "train_rmse",
                torch.sqrt(self.train_mse.compute()),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "train_mae",
                self.train_mae,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "train_r2",
                self.train_r2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        else:
            y_true = y_true.int()
            self.train_auc.update(torch.sigmoid(y_pred), y_true)
            self.train_f1.update(torch.sigmoid(y_pred), y_true)
            self.log(
                "train_auc",
                self.train_auc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "train_f1",
                self.train_f1,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        f_g, mfg, num_features, y_true = batch
        y_pred, _ = self(f_g, mfg, num_features)
        loss = self.criterion(y_pred, y_true)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.regression:
            self.val_mse.update(y_pred, y_true)
            self.val_mae.update(y_pred, y_true)
            self.val_r2.update(y_pred, y_true)
            self.log(
                "val_rmse",
                torch.sqrt(self.val_mse.compute()),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )
            self.log(
                "val_r2",
                self.val_r2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        else:
            y_true = y_true.int()
            self.val_auc.update(torch.sigmoid(y_pred), y_true)
            self.val_f1.update(torch.sigmoid(y_pred), y_true)
            self.log(
                "val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )
            self.log(
                "val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )

    def test_step(self, batch, batch_idx):
        f_g, mfg, num_features, y_true = batch
        y_pred, _ = self(f_g, mfg, num_features)
        loss = self.criterion(y_pred, y_true)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.regression:
            self.test_mse.update(y_pred, y_true)
            self.test_mae.update(y_pred, y_true)
            self.test_r2.update(y_pred, y_true)
            self.log(
                "test_rmse",
                torch.sqrt(self.test_mse.compute()),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "test_mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )
            self.log(
                "test_r2",
                self.test_r2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        else:
            y_true = y_true.int()
            self.test_auc.update(torch.sigmoid(y_pred), y_true)
            self.test_f1.update(torch.sigmoid(y_pred), y_true)
            self.log(
                "test_auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )
            self.log(
                "test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )


class FGRPretrainLightning(LightningModule):
    """Lightning module for pretraining"""

    def __init__(
        self,
        fg_input_dim: int,
        mfg_input_dim: int,
        method: str,
        hidden_dims: List[int],
        bottleneck_dim: int,
        lr: float,
        weight_decay: float,
        max_lr: float,
        **kwargs
    ):
        """Initialize the FGR Pretrain model

        Args:
            fg_input_dim (int): Input dimension for FG
            mfg_input_dim (int): Input dimension for MFG
            method (str): Representation method to train
            hidden_dims (Tuple[int]): Dimensions for each layer
            bottleneck_dim (int): Dimension of bottleneck layer
            lr (float): Learning rate for optimizer
            weight_decay (float): Weight decay for optimizer
            max_lr (float): Maximum learning rate for scheduler
        """

        super().__init__()
        self.save_hyperparameters()
        self.net = FGRPretrainModel(
            fg_input_dim,
            mfg_input_dim,
            hidden_dims,
            bottleneck_dim,
            method,
        )
        self.l_r = lr
        self.method = method
        self.weight_decay = weight_decay
        self.max_lr = max_lr
        self.recon_loss = nn.BCEWithLogitsLoss()

    def forward(self, fgr):
        preds = self.net(fgr)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.l_r, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.max_lr, total_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        fgr = batch
        _, recon = self(fgr)
        loss = self.recon_loss(recon, fgr)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        fgr = batch
        _, recon = self(fgr)
        loss = self.recon_loss(recon, fgr)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        fgr = batch
        _, recon = self(fgr)
        loss = self.recon_loss(recon, fgr)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
