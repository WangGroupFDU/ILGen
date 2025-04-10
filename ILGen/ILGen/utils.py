import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from more_itertools import chunked
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
import networkx as nx
import pandas as pd
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem import rdmolops
from rdkit.Chem.Draw import IPythonConsole
import seaborn as sns
import re
from rdkit.Chem import AddHs
from rdkit.Chem import AllChem
import os
from openbabel import openbabel, pybel
from rdkit.Chem.rdmolfiles import MolToSmiles
from rdkit.Chem import RemoveHs
import subprocess
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from openbabel import openbabel
from openpyxl import Workbook
import time
from openbabel import pybel
import shutil
import glob
from IPython.display import display
from rdkit.Chem import Draw
from openbabel import openbabel as ob
from matplotlib.ticker import MaxNLocator
import json
from itertools import combinations
from rdkit import DataStructs
import matplotlib.pyplot as plt
import itertools
from itertools import product
#from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import Draw
from IPython.display import display
import random
from rdkit.Chem import RWMol, Atom
from rdkit.Chem.rdchem import AtomValenceException, KekulizeException
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import MaxNLocator
from sklearn.random_projection import GaussianRandomProjection
from joblib import Parallel, delayed
from openpyxl import load_workbook
from sklearn.decomposition import PCA
from rdkit import RDLogger
import warnings
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils import shuffle
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import numpy as np
from collections import Counter
import re 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem 
import os
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

def classify_cation(df, *col_names):

    smarts_to_type = {
        'c1c[nH+]c[nH]1': 'imidazolium',
        '[*]n1cc[n+]([*])c1': 'imidazolium',
        '[*]n1cc[nH+]c1': 'imidazolium',
        '[*]c1[nH]cc[n+]1[*]': 'imidazolium',
        '[*][n+]1cc[nH]c1': 'imidazolium',
        '[*]c1[nH]cc[nH+]1': 'imidazolium',
        '[*][n+]1c(*)cc(*)n1[*]': 'pyrazolium',
        '[*][n+]1ccccc1': 'pyridinium',
        '[*]c1cccc[nH+]1': 'pyridinium',
        'c1cc[nH+]cc1': 'pyridinium',
        '[*][n+]1ccccn1': 'pyrazinium',
        '[*][N+]1([*])CCCC1': 'pyrrolidium',
        '[*][N+]1([*])CCCCC1': 'piperidinium',
        '[*][N+]([*])([*])[*]': 'quaternaryammonium',
        '[*][NH+][*][*]': 'quaternaryammonium',
        '[*][NH2+][*]': 'quaternaryammonium',
        '[*][NH3+]': 'quaternaryammonium',
        '[NH4+]': 'quaternaryammonium',
        '[*][NH+][*]': 'tertiaryammonium',
        '[*][N+][*]': 'secondaryammonium',
        '[*][P+]([*])([*])[*]': 'tetraphosphonium ',
        '[*][PH+]([*])[*]': 'tetraphosphonium ',
        '[*][PH2+][*]': 'tetraphosphonium ',
        '[*][PH3+]': 'tetraphosphonium ',
        '[*][S+]([*])[*]': 'sulfonium',
        '[*]N([*])C(N([*])[*])=[N+]([*])[*]': 'guanidinium',
        '[*]N([*])C(N([*])[*])=[NH2+]': 'guanidinium',
        'NC(N)=[NH2+]': 'guanidinium',
        '[*]n1cc[n+]([*])n1': 'triazolium',
        '[*]n1cn[n+](*)c1': 'triazolium',
        '[*]n1c[n+](*)cn1': 'triazolium',
        '[*][n+]1cnc[nH]1': 'triazolium',
        '[*]n1cn[nH+]c1': 'triazolium',
        'c1nc[nH+][nH]1': 'triazolium',
        '[*]c1nc[nH+]n1[*]': 'triazolium',
        '[*]c1nc(*)[nH+][nH]1': 'triazolium',
        '[*]C1=[NH+]N(*)NN1[*]': 'tetrazolium',
        '[*]c1n(*)nn[n+]1[*]': 'tetrazolium',
        'Cn1n[nH+]n(*)c1[*]': 'tetrazolium',
        '[*]c1[nH]nn[n+]1[*]': 'tetrazolium',
        'Cn1n[nH+]n(*)c1=N': 'tetrazolium',
        '[*][O+]1CCCCC1': 'pyrylium',
        '[*][n+]1ccsc1': 'thiazolium',
        'c1ccc2sc[nH+]c2c1': 'benzothiazole',
        
    }

    smarts_patterns = {smarts: Chem.MolFromSmarts(smarts) for smarts in smarts_to_type.keys()}

    def match_type(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        for smarts, compound_type in smarts_patterns.items():
            if mol.HasSubstructMatch(compound_type):
                return smarts_to_type[smarts]
        return 'unknown'

    for col_name in col_names:
        df.loc[:, f'Cation_{col_name}_type'] = df[col_name].apply(match_type)
        type_counts = df[f'Cation_{col_name}_type'].value_counts()
        print(f"Type counts for column '{col_name}':")
        print(type_counts)
        print()

        unknown_smiles = df[df[f'Cation_{col_name}_type'] == 'unknown'][col_name]
        if not unknown_smiles.empty:
            print(f"Unknown SMILES for column '{col_name}':")
            print(unknown_smiles)
            print() 
        else:
            print(f"No unknown SMILES found in column '{col_name}'.\n")
        
    return df

def classify_anion(df, *col_names):

    smarts_to_type = {
        '[*][B-](F)(F)F': 'fluoroborate',    
        '[B-](F)(F)': 'fluoroborate',    
        '[B-](F)': 'fluoroborate',    
        '[*]C(=O)[O-]': 'acetate',       
        '[*][N-][*]': 'amide anion',         
        '[*][n-][*]': 'amide anion',         
        '[*]S(=O)(=O)[O-]': 'sulfonate',      
        '[*]P(=O)([O-])[*]': 'phosphate',    
        '[*][PH](=O)[O-]': 'phosphate',   
        '[*]OP(C)(=O)[O-]': 'phosphate',    
        '[*]O[B-](O[*])(O[*])O[*]': 'borate', 
        '[*]OOB([O-])OO[*]': 'borate',  
        '[*][O-]': 'Oxygen anion',       
        '[*][o-]': 'Oxygen anion',    
    }

    smarts_patterns = {smarts: Chem.MolFromSmarts(smarts) for smarts in smarts_to_type.keys()}

    def match_type(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        for smarts, pattern in smarts_patterns.items():
            if mol.HasSubstructMatch(pattern):
                return smarts_to_type[smarts]
        return 'unknown'

    for col_name in col_names:
        df.loc[:, f'Anion_{col_name}_type'] = df[col_name].apply(match_type)
        
        type_counts = df[f'Anion_{col_name}_type'].value_counts()
        print(type_counts)

        unknown_smiles = df[df[f'Anion_{col_name}_type'] == 'unknown'][col_name]
        if not unknown_smiles.empty:
            print(f"'{col_name}' unknown_smiles: ")
            print(unknown_smiles)
        else:
            pass
        
    return df

def stratified_sampling(df, col, sample_size):
    class_counts = df[col].value_counts()
    n_classes = class_counts.size

    samples_per_class = sample_size // n_classes
    sampled_data = []

    deficit = 0

    for class_label, count in class_counts.items():
        if count >= samples_per_class:
            sampled_data.append(df[df[col] == class_label].sample(n=samples_per_class, random_state=0))
        else:
            sampled_data.append(df[df[col] == class_label])
            deficit += samples_per_class - count

    sampled_df = pd.concat(sampled_data)

    if deficit > 0:
        remaining_df = df[~df[col].isin(sampled_df[col])]
        if not remaining_df.empty:
            additional_samples = resample(remaining_df, n_samples=deficit, random_state=0, replace=False)
            sampled_df = pd.concat([sampled_df, additional_samples])
        else:
            additional_samples = resample(sampled_df, n_samples=deficit, random_state=0, replace=True)
            sampled_df = pd.concat([sampled_df, additional_samples])

    return sampled_df

def combine_ion2IL(cation_df, anion_df, cation_limit=5000, anion_limit=3000, seed=1):

    if 'Name' not in cation_df.columns or 'SMILES' not in cation_df.columns:
        raise ValueError("cation_df must contain 'Name' and 'SMILES' ")
    cation_df = cation_df[['Name', 'SMILES']]

    if 'Name' not in anion_df.columns or 'SMILES' not in anion_df.columns:
        raise ValueError("anion_df must contain 'Name' 和 'SMILES' ")
    anion_df = anion_df[['Name', 'SMILES']]

    cation_df = classify_cation(cation_df, "SMILES")
    anion_df = classify_anion(anion_df, "SMILES")
    
    if len(cation_df) > cation_limit:
        cation_df = stratified_sampling(cation_df, col="Cation_SMILES_type", sample_size=cation_limit)
    
    if len(anion_df) > anion_limit:
        anion_df = stratified_sampling(anion_df, col="Anion_SMILES_type", sample_size=anion_limit)

    combinations = []

    for _, cation_row in cation_df.iterrows():
        for _, anion_row in anion_df.iterrows():
            # 基础信息组合
            combined_row = {
                "Name": f"{cation_row['Name']}.{anion_row['Name']}",
                "SMILES": f"{cation_row['SMILES']}.{anion_row['SMILES']}",
                "Anion_Name": anion_row['Name'],
                "Cation_Name": cation_row['Name'],
                "Anion_SMILES": anion_row['SMILES'],
                "Cation_SMILES": cation_row['SMILES'],
                "Cation_SMILES_type": cation_row['Cation_SMILES_type'],
                "Anion_SMILES_type": anion_row['Anion_SMILES_type'],
            }

            combinations.append(combined_row)

    combined_df = pd.DataFrame(combinations)
    
    return combined_df