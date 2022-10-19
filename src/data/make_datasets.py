"""Module for making datasets for training and testing."""

from typing import Callable, Dict, List
from rdkit import RDLogger
import numpy as np
from deepchem.data.datasets import DiskDataset
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from utils import util_funcs


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class FGRDataset(Dataset):
    """Pytorch dataset for training and testing different models"""

    def __init__(
        self,
        data: DiskDataset,
        fgroups_list: List[str],
        tokenizer: Tokenizer,
        descriptor_funcs: Dict[str, Callable],
    ) -> None:
        """Initiliaze dataset with arguments

        Args:
            data (DiskDataset): Deepchem dataset containing SMILES and labels
            fgroups_list (List[str]): List of functional groups
            tokenizer (Tokenizer): Pretrained tokenizer
            descriptor_funcs (Dict[str, Callable]): RDKit descriptor dictionary
        """
        self.mols = data.X
        self.labels = data.y
        self.smiles = data.ids
        self.fgroups_list = fgroups_list
        self.tokenizer = tokenizer
        self.descriptor_funcs = descriptor_funcs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mol = self.mols[idx]
        smile = self.smiles[idx]
        target = self.labels[idx]
        f_g, mfg = util_funcs.smiles2vector_fgr(smile, self.tokenizer, self.fgroups_list)
        num_features = np.asarray(
            [self.descriptor_funcs[key](mol) for key in self.descriptor_funcs.keys()]
        )
        return f_g, mfg, num_features, target


class FGRPretrainDataset(Dataset):
    """Pytorch dataset for pretraining autoencoder"""

    def __init__(self, smiles: List[str], fgroups_list: List[str], tokenizer: Tokenizer) -> None:
        """Initialize dataset with arguments

        Args:
            smiles (List[str]): List of SMILES strings
            fgroups_list (List[str]): List of functional groups
            tokenizer (Tokenizer): Pretrained Tokenizer
        """
        self.smiles = smiles
        self.fgroups_list = fgroups_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        f_g, mfg = util_funcs.smiles2vector_fgr(smile, self.tokenizer, self.fgroups_list)
        return f_g, mfg
