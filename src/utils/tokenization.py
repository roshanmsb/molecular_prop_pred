import dask.dataframe as dd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]"], min_frequency=1500, show_progress=True)  # type: ignore
tokenizer.pre_tokenizer = Whitespace()  # type: ignore


df = dd.read_parquet("../datasets/processed/pubchem_proc")
smiles = df["check"]
smiles = smiles.compute()

tokenizer.train_from_iterator(smiles, trainer=trainer)
tokenizer.save("../datasets/processed/tokenizer_bpe.json")
