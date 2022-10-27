"""Module for making datasets for training and testing."""

from typing import Callable, Dict, List
from rdkit import RDLogger
from rdkit.Chem.rdmolfiles import MolFromSmiles
import numpy as np
from deepchem.data.datasets import DiskDataset
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from src.utils import util_funcs
from molvs import standardize_smiles

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
        method: str,
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
        self.method = method

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        smile = standardize_smiles(self.smiles[idx])
        target = self.labels[idx]
        if self.method == "FG":
            f_g = util_funcs.smiles2vector_fg(smile, self.fgroups_list)
            return f_g, target
        elif self.method == "MFG":
            mfg = util_funcs.smiles2vector_mfg(smile, self.tokenizer)
            return mfg, target
        elif self.method == "FGR":
            f_g = util_funcs.smiles2vector_fg(smile, self.fgroups_list)
            mfg = util_funcs.smiles2vector_mfg(smile, self.tokenizer)
            return np.concatenate((f_g, mfg)), target
        elif self.method == "FGR_desc":
            mol = MolFromSmiles(smile)
            f_g = util_funcs.smiles2vector_fg(smile, self.fgroups_list)
            mfg = util_funcs.smiles2vector_mfg(smile, self.tokenizer)
            num_features = np.asarray(
                [self.descriptor_funcs[key](mol) for key in self.descriptor_funcs.keys()]
            )
            return np.concatenate((f_g, mfg)), num_features, target
        else:
            raise ValueError("Method not supported")


class FGRPretrainDataset(Dataset):
    """Pytorch dataset for pretraining autoencoder"""

    def __init__(
        self, smiles: List[str], fgroups_list: List[str], tokenizer: Tokenizer, method: str
    ) -> None:
        """Initialize dataset with arguments

        Args:
            smiles (List[str]): List of SMILES strings
            fgroups_list (List[str]): List of functional groups
            tokenizer (Tokenizer): Pretrained Tokenizer
        """
        self.smiles = smiles
        self.fgroups_list = fgroups_list
        self.tokenizer = tokenizer
        self.method = method

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = standardize_smiles(self.smiles[idx])
        if self.method == "FG":
            f_g = util_funcs.smiles2vector_fg(smile, self.fgroups_list)
            return f_g
        elif self.method == "MFG":
            mfg = util_funcs.smiles2vector_mfg(smile, self.tokenizer)
            return mfg
        elif self.method == "FGR":
            f_g = util_funcs.smiles2vector_fg(smile, self.fgroups_list)
            mfg = util_funcs.smiles2vector_mfg(smile, self.tokenizer)
            return np.concatenate((f_g, mfg))
        else:
            raise ValueError("Method not supported for pretraining")
