from dask.diagnostics import ProgressBar
from rdkit.Chem.rdmolfiles import MolFromSmiles
import dask.dataframe as dd


df = dd.read_csv("../datasets/processed/" + "pubchem/pubchem_data_*.csv")
print("Reading SMILES")
smiles_mol = df["SMILES"].map(lambda x: MolFromSmiles(x))
df['mol'] = smiles_mol
df = df.dropna()
with ProgressBar():
    print("Converting SMILES to MOL")
    df.compute()
    df.to_csv('../datasets/processed/pubchem_proc/pubchem_data_*.csv', index=False)
