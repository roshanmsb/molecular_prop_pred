from src.data.data_module import FGRPretrainDataModule
from tqdm.auto import tqdm

data_module = FGRPretrainDataModule("datasets/processed/", "pubchem", 32, 32, True, method="FG")
data_module.setup()
lo = data_module.train_dataloader()
for batch in tqdm(lo):
    pass
