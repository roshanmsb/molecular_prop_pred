from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def make_encoder_decoder(
    input_dim: int, hidden_dims: List[int], bottleneck_dim: int,
) -> Tuple[nn.modules.container.Sequential, nn.modules.container.Sequential]:
    """Function for creating encoder and decoder models

    Args:
        input_dim (int): Input dimension for encoder
        hidden_dims (List[int]): Dimensions for each layer
        bottleneck_dim (int): Dimension of bottleneck layer

    Returns:
        Tuple[nn.modules.container.Sequential]: Encoder and decoder models
    """

    encoder_layers = []
    decoder_layers = []
    output_dim = input_dim
    dec_shape = bottleneck_dim
    for enc_dim in hidden_dims:
        encoder_layers.extend([nn.Linear(input_dim, enc_dim), nn.SiLU()])
        input_dim = enc_dim

    encoder_layers.append(nn.Linear(input_dim, bottleneck_dim))

    dec_dims = list(reversed(hidden_dims))
    for dec_dim in dec_dims:
        decoder_layers.extend([nn.Linear(dec_shape, dec_dim), nn.SiLU()])
        dec_shape = dec_dim

    decoder_layers.append(nn.Linear(dec_shape, output_dim))

    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


class FGRModel(nn.Module):
    """Pytorch model for FG based representation"""

    def __init__(
        self,
        fg_input_dim: int,
        mfg_input_dim: int,
        num_feat_dim: int,
        hidden_dims: List[int],
        bottleneck_dim: int,
        output_dims: List[int],
        num_tasks: int,
        dropout: float,
        method: str,
    ) -> None:
        """Initialize Pytorch model

        Args:
            fg_input_dim (int): Input dimension for FG
            mfg_input_dim (int): Input dimension for MFG
            num_feat_dim (int): Input dimension for RDKit features
            hidden_dims (List[int]): Dimensions for each layer
            bottleneck_dim (int): Dimension of bottleneck layer
            output_dims (List[int]): Dimensions for each layer in predictor
            num_tasks (int): Number of tasks for each dataset
            dropout (float): Dropout for each layer
            method (str): Representation method to train
        """
        super().__init__()

        self.method = method
        if self.method == "FG":
            input_dim = fg_input_dim
        elif self.method == "MFG":
            input_dim = mfg_input_dim
        elif self.method == "FGR" or self.method == "FGR_desc":
            input_dim = fg_input_dim + mfg_input_dim
        else:
            raise ValueError("Method not supported")
        self.encoder, self.decoder = make_encoder_decoder(input_dim, hidden_dims, bottleneck_dim)

        self.dropout = nn.Dropout(dropout)
        self.predict_out_dim = num_tasks

        if self.method == "FGR_desc":
            fcn_input_dim = bottleneck_dim + num_feat_dim
        else:
            fcn_input_dim = bottleneck_dim

        layers = []
        for output_dim in output_dims:
            layers.extend(
                [nn.Linear(fcn_input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.SiLU()]
            )
            fcn_input_dim = output_dim

        layers.extend([self.dropout, nn.Linear(fcn_input_dim, num_tasks)])

        self.predictor = nn.Sequential(*layers)

    def forward(self, x, num_feat=None):
        """Perform forward pass"""

        z_d = self.encoder(self.dropout(x))
        v_d_hat = self.decoder(z_d)
        if self.method == "FGR_desc":
            num_feat = torch.nan_to_num(num_feat, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore
            z_d = torch.cat([z_d, F.normalize(num_feat)], dim=1)
        output = self.predictor(z_d)
        return output, v_d_hat


class FGRPretrainModel(nn.Module):
    """Pytorch model for Pretraining FG based representation"""

    def __init__(
        self,
        fg_input_dim: int,
        mfg_input_dim: int,
        hidden_dims: List[int],
        bottleneck_dim: int,
        method: str,
    ) -> None:
        """Initialize Pytorch model

        Args:
            fg_input_dim (int): Input dimension for FG
            mfg_input_dim (int): Input dimension for MFG
            hidden_dims (List[int]): Dimensions for each layer
            bottleneck_dim (int): Dimension of bottleneck layer
            method (str): Representation method to train
        """
        super().__init__()

        self.method = method
        if self.method == "FG":
            input_dim = fg_input_dim
        elif self.method == "MFG":
            input_dim = mfg_input_dim
        elif self.method == "FGR":
            input_dim = fg_input_dim + mfg_input_dim
        else:
            raise ValueError("Method not supported")
        self.encoder, self.decoder = make_encoder_decoder(input_dim, hidden_dims, bottleneck_dim)

    def forward(self, x):
        """Perform forward pass"""

        z_d = self.encoder(x)
        v_d_hat = self.decoder(z_d)
        return z_d, v_d_hat
