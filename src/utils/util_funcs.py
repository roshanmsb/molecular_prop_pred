"""Module for utility functions"""
from typing import List, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromSmiles
import pandas as pd
from deepchem.data import DiskDataset
from rdkit import RDLogger
import tokenizers


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def smiles2index(s_1: str, tokenizer: tokenizers.Tokenizer) -> List[int]:
    """Tokenize a SMILES string

    Args:
        s_1 (str): SMILES string
        tokenizer (tokenizers.Tokenizer): Pretrained tokenizer

    Returns:
        List[int]: List of tokens
    """
    return tokenizer.encode(str(s_1)).ids


def index2multi_hot_fg(molecule: Chem.rdchem.Mol, fgroups_list: List[str]) -> np.ndarray:
    """Generate functional group representation

    Args:
        molecule (Chem.rdchem.Mol): Rdkit molecule from SMILES string
        fgroups_list (List[str]): List of SMARTS strings for functional groups

    Returns:
        List[int]: One hot encoding of functional groups
    """
    v_1 = np.zeros(len(fgroups_list))
    for idx, f_g in enumerate(fgroups_list):
        if molecule.HasSubstructMatch(f_g):
            v_1[idx] = 1
    return v_1


def smiles2vector_fgr(
    s_1: str, tokenizer: tokenizers.Tokenizer, fgroups_list: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Functional Groups (FG) and Mined Functional Groups (MFG)

    Args:
        s_1 (str): SMILES string
        tokenizer (tokenizers.Tokenizer): Pretrained tokenizer
        fgroups_list (List[str]): List of SMARTS strings for functional groups

    Returns:
        Tuple[List[int],List[int]]: FG and MFG
    """
    i_1 = smiles2index(s_1, tokenizer)
    mfg = np.zeros(tokenizer.get_vocab_size())
    mfg[i_1] = 1
    molecule = MolFromSmiles(s_1)
    f_g = index2multi_hot_fg(molecule, fgroups_list)
    return f_g, mfg


def get_weights(labels: np.ndarray) -> np.ndarray:
    """Calculate weights for all samples

    Args:
        labels (np.ndarray): Labels for each task

    Returns:
        np.ndarray: Sample weights
    """
    class_sample_count = np.unique(labels, return_counts=True)[1]
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in labels])
    return samples_weight


def load_dataset(task_name: str) -> DiskDataset:
    """Load local datasets in deepchem format

    Args:
        task_name (str): Name of task

    Returns:
        DiskDataset: Deepchem compatible dataset
    """
    d_f = pd.read_csv(f"../datasets/processed/{task_name}.csv")
    ids = d_f["SMILES"]
    labels = np.expand_dims(d_f["Target"], 1)
    mols = [MolFromSmiles(smiles) for smiles in ids]
    data = DiskDataset.from_numpy(X=mols, y=labels.astype(float), ids=ids, w=get_weights(labels))
    data.tasks = np.asarray([task_name])
    return data
