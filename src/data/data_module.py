"""Module for lightning datamodules"""

import dask.dataframe as dd
import deepchem.molnet as dcm
import pandas as pd
from deepchem.splits.splitters import RandomStratifiedSplitter, ScaffoldSplitter
from lightning.pytorch import LightningDataModule
from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolfiles import MolFromSmarts
from tokenizers import Tokenizer
from tokenizers.models import BPE
from torch.utils.data import DataLoader, Subset

from src.data.datasets import FGRDataset, FGRPretrainDataset
from src.utils import util_funcs


class FGRDataModule(LightningDataModule):
    """Lightning Datamodule for loading training and testing"""

    def __init__(
        self,
        root: str,
        task_name: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        split_type: str,
        num_folds: int,
        fold_index: int,
        method: str,
    ) -> None:
        """Initialize lightning datamodule for training and testing

        Args:
            root (str): Root data folder
            task_name (str): Task for loading data
            batch_size (int): Batch size
            num_workers (int): Number of workers for data loading
            pin_memory (bool): Save data in memory
            split_type (str): Type of splitting data
            num_folds (int): Number of independent folds
            fold_index (int): Fold number to load data
            method (str): Method for training
        """
        super().__init__()

        self.root = root
        self.task_name = task_name
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.split_type = split_type
        self.num_folds = num_folds
        self.fold_index = fold_index
        self.method = method

    def setup(self, stage=None):
        try:
            load_fn = getattr(dcm, "load_%s" % self.task_name)
            data = load_fn(featurizer="raw", splitter=None)[1][0]
        except AttributeError:
            data = util_funcs.load_dataset(self.root, self.task_name)
        fgroups = pd.read_csv(self.root + "fg.csv")["SMARTS"].tolist()
        self.fgroups_list = [MolFromSmarts(x) for x in fgroups]
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]")).from_file(
            self.root + "tokenizer_bpe.json"
        )
        self.descriptor_funcs = {name: func for name, func in Descriptors.descList}

        if self.split_type == "scaffold":
            splitter = ScaffoldSplitter()
            self.splits = []
            for i in range(self.num_folds):
                new_data = data.complete_shuffle()
                self.splits.append((new_data, splitter.split(dataset=new_data, seed=i)))
            self.dataset = FGRDataset(
                self.splits[self.fold_index][0],
                self.fgroups_list,
                self.tokenizer,
                self.descriptor_funcs,
                self.method,
            )
            self.train_ind, self.val_ind, self.test_ind = self.splits[self.fold_index][1]
            assert len(set(self.train_ind) & set(self.val_ind) & set(self.test_ind)) == 0
        else:
            splitter = RandomStratifiedSplitter()
            self.splits = [
                splitter.split(dataset=data, seed=fold_num) for fold_num in range(self.num_folds)
            ]
            self.dataset = FGRDataset(
                data, self.fgroups_list, self.tokenizer, self.descriptor_funcs, self.method
            )
            self.train_ind, self.val_ind, self.test_ind = self.splits[self.fold_index]
            assert len(set(self.train_ind) & set(self.val_ind) & set(self.test_ind)) == 0
        self.train_fold = Subset(self.dataset, self.train_ind)  # type: ignore
        self.val_fold = Subset(self.dataset, self.val_ind)  # type: ignore
        self.test_fold = Subset(self.dataset, self.test_ind)  # type: ignore

    def train_dataloader(self):
        loader = DataLoader(
            self.train_fold,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_fold,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_fold,  # type: ignore
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader


class FGRPretrainDataModule(LightningDataModule):
    """Lightning Datamodule for pretraining"""

    def __init__(
        self,
        root: str,
        dataset_name: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        method: str,
    ) -> None:
        """Initialize lightning datamodule for pretraining

        Args:
            root (str): Root data folder
            batch_size (int): Batch size
            num_workers (int): Number of workers for data loading
            pin_memory (bool): Save data in memory
            method (str): Method for training
        """
        super().__init__()

        self.root = root
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.method = method
        self.dataset = dataset_name

    def setup(self, stage=None):
        df = dd.read_parquet(self.root + self.dataset)["SMILES"]
        self.train, self.valid = df.random_split((0.9, 0.1), random_state=123)  # type: ignore
        self.train = self.train.compute()
        self.valid = self.valid.compute()
        fgroups = pd.read_csv(self.root + "fg.csv")["SMARTS"].tolist()
        self.fgroups_list = [MolFromSmarts(x) for x in fgroups]
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]")).from_file(
            self.root + "tokenizer_bpe.json"
        )
        self.train_fold = FGRPretrainDataset(
            self.train, self.fgroups_list, self.tokenizer, self.method
        )

        self.val_fold = FGRPretrainDataset(
            self.valid, self.fgroups_list, self.tokenizer, self.method
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_fold,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_fold,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader
