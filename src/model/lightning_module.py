from typing import List

import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import BaseFinetuning
from torch import nn
from torchvision.ops import sigmoid_focal_loss

from src.model.network import FGRModel, FGRPretrainModel


class FGRLightning(LightningModule):
    """Lightning module for training"""

    def __init__(
        self,
        fg_input_dim: int,
        mfg_input_dim: int,
        num_feat_dim: int,
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
            num_feat_dim (int): Input dimension for RDKit features
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
            num_feat_dim,
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
            self.train_auc = torchmetrics.AUROC(task = 'binary',num_classes=num_tasks)
            self.val_auc = torchmetrics.AUROC(task = 'binary',num_classes=num_tasks)
            self.test_auc = torchmetrics.AUROC(task = 'binary',num_classes=num_tasks)
            self.train_f1 = torchmetrics.F1Score(task = 'binary',num_classes=num_tasks)
            self.val_f1 = torchmetrics.F1Score(task = 'binary',num_classes=num_tasks)
            self.test_f1 = torchmetrics.F1Score(task = 'binary',num_classes=num_tasks)

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
            optimizer, max_lr=self.max_lr, total_steps=self.trainer.estimated_stepping_batches) # type: ignore
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        if self.method == "FGR_desc":
            fgr, num_features, y_true = batch
            y_pred, recon = self(fgr, num_features)
        else:
            fgr, y_true = batch
            y_pred, recon = self(fgr)

        loss_r_pre = sigmoid_focal_loss(recon, fgr, reduction="mean")
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
                "train_r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=True, logger=True,
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
                "train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True,
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
                "val_r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True, logger=True,
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
                "test_r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=True, logger=True,
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
            hidden_dims (List[int]): Dimensions for each layer
            lr (float): Learning rate for optimizer
            weight_decay (float): Weight decay for optimizer
            max_lr (float): Maximum learning rate for scheduler
        """

        super().__init__()
        self.save_hyperparameters()
        self.net = FGRPretrainModel(
            fg_input_dim, mfg_input_dim, hidden_dims, bottleneck_dim, method,
        )
        self.l_r = lr
        self.method = method
        self.weight_decay = weight_decay
        self.max_lr = max_lr

    def forward(self, fgr):
        preds = self.net(fgr)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.l_r, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(  
            optimizer, max_lr=self.max_lr, total_steps=self.trainer.estimated_stepping_batches # type: ignore
        ) 
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        fgr = batch
        _, recon = self(fgr)
        loss = sigmoid_focal_loss(recon, fgr, reduction="mean")
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        fgr = batch
        _, recon = self(fgr)
        loss = sigmoid_focal_loss(recon, fgr, reduction="mean")
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )


class Finetuning(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=25):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing `feature_extractor`
        self.freeze(pl_module.encoder)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            print("Unfreezing encoder")
            self.unfreeze_and_add_param_group(
                modules=pl_module.encoder, optimizer=optimizer, train_bn=True,
            )


class FGRFinetuneLightning(LightningModule):
    """Lightning module for finetuning"""

    def __init__(
        self,
        checkpoint_path: str,
        fg_input_dim: int,
        mfg_input_dim: int,
        num_feat_dim: int,
        bottleneck_dim: int,
        num_tasks: int,
        method: str,
        regression: bool,
        output_dims: List[int],
        dropout: float,
        lr: float,
        weight_decay: float,
        max_lr: float,
        **kwargs
    ):
        """Initialize the FGR model

        Args:
            checkpoint_path (str): Path to checkpoint
            fg_input_dim (int): Input dimension for FG
            mfg_input_dim (int): Input dimension for MFG
            num_feat_dim (int): Input dimension for RDKit features
            bottleneck_dim (int): Bottleneck dimension
            num_tasks (int): Number of tasks for each dataset
            method (str): Representation method to train
            regression (bool):Whether the task is regression or classification
            output_dims (List[int]): Dimensions for each layer in predictor
            dropout (float): Dropout for each layer
            lr (float): Learning rate for optimizer
            weight_decay (float): Weight decay for optimizer
            max_lr (float): Maximum learning rate for scheduler
        """

        super().__init__()
        self.save_hyperparameters()
        self.encoder = self.load_model()

        if method == "FGR_desc":
            fcn_input_dim = bottleneck_dim + num_feat_dim
        else:
            fcn_input_dim = bottleneck_dim

        self.dropout = nn.Dropout(dropout)
        layers = []
        for output_dim in output_dims:
            layers.extend(
                [nn.Linear(fcn_input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.SiLU()]
            )
            fcn_input_dim = output_dim

        layers.extend([self.dropout, nn.Linear(fcn_input_dim, num_tasks)])

        self.predictor = nn.Sequential(*layers)
        self.l_r = lr
        self.method = method
        self.regression = regression
        self.weight_decay = weight_decay
        self.max_lr = max_lr
        if self.regression:
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

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
            self.train_auc = torchmetrics.AUROC(task = 'binary',num_classes=num_tasks)
            self.val_auc = torchmetrics.AUROC(task = 'binary',num_classes=num_tasks)
            self.test_auc = torchmetrics.AUROC(task = 'binary',num_classes=num_tasks)
            self.train_f1 = torchmetrics.F1Score(task = 'binary',num_classes=num_tasks)
            self.val_f1 = torchmetrics.F1Score(task = 'binary',num_classes=num_tasks)
            self.test_f1 = torchmetrics.F1Score(task = 'binary',num_classes=num_tasks)

    def load_model(self):
        model = FGRPretrainLightning(2786, 3000, "FGR", [256], 64, 9e-3, 0.12, 0.02)
        checkpoint = torch.load("checkpoints/pretrain.ckpt")
        model.load_state_dict(checkpoint["state_dict"])
        return model.net.encoder

    def forward(self, fgr=None, num_feat=None):
        if self.method != "FGR_desc":
            z_d = self.encoder(fgr)
            preds = self.predictor(z_d)
        else:
            num_feat = torch.nan_to_num(num_feat, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore
            z_d = self.encoder(fgr)
            num_feat = num_feat.half()
            z_d = torch.cat([z_d, F.normalize(num_feat)], dim=1)  # type: ignore
            preds = self.predictor(z_d)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.l_r,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=1e-2)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        if self.method == "FGR_desc":
            fgr, num_features, y_true = batch
            y_pred = self(fgr, num_features)
        else:
            fgr, y_true = batch
            y_pred = self(fgr)
        if self.regression:
            y_true = y_true.float()
        criterion_loss = self.criterion(y_pred, y_true)
        loss = criterion_loss
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
                "train_r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=True, logger=True,
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
                "train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        if self.method == "FGR_desc":
            fgr, num_features, y_true = batch
            y_pred = self(fgr, num_features)
        else:
            fgr, y_true = batch
            y_pred = self(fgr)
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
                "val_r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True, logger=True,
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
        if self.method == "FGR_desc":
            fgr, num_features, y_true = batch
            y_pred = self(fgr, num_features)
        else:
            fgr, y_true = batch
            y_pred = self(fgr)
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
                "test_r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=True, logger=True,
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
