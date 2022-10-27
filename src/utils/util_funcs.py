"""Module for utility functions"""
from typing import List
import numpy as np
from rdkit.Chem.rdmolfiles import MolFromSmiles
import pandas as pd
from deepchem.data import DiskDataset
from rdkit import RDLogger
import tokenizers


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def smiles2vector_fg(s_1: str, fgroups_list: List[str]) -> np.ndarray:
    molecule = MolFromSmiles(s_1)
    v_1 = np.zeros(len(fgroups_list), dtype=np.float32)
    for idx, f_g in enumerate(fgroups_list):
        if molecule.HasSubstructMatch(f_g):
            v_1[idx] = 1
    return v_1


def smiles2vector_mfg(s_1: str, tokenizer: tokenizers.Tokenizer) -> np.ndarray:
    i_1 = tokenizer.encode(s_1).ids
    mfg = np.zeros(tokenizer.get_vocab_size(), dtype=np.float32)
    mfg[i_1] = 1
    return mfg


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
