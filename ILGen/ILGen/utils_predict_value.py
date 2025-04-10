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
from MLPModel import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 2048
hidden_sizes = [256, 64]
output_size = 1
model_conductivity_MLP = MLP(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size).to(device) 

IL_ECW_xgb_model_path = 'model/IL_ECW_xgb_model.joblib'
IL_ECW_xgb_model = joblib.load(IL_ECW_xgb_model_path)

Tm_xgb_model_path = 'model/Tm_xgb_model.joblib'
Tm_xgb_model = joblib.load(Tm_xgb_model_path)

MLP_model_path = 'model/conductivity_MLP_model.pt'
state_dict = torch.load(MLP_model_path, map_location=device)
model_conductivity_MLP.load_state_dict(state_dict)


def add_hydrogens_to_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles) 
    mol = Chem.AddHs(mol)
    smiles_with_h = Chem.MolToSmiles(mol)
    return smiles_with_h

def extract_features_targets(data_list, feature = "fp"):
    X = []

    for data in data_list:
        if feature == "2Ddescriptors":
            moldescriptor = data.moldescriptor.numpy().flatten()
            X.append(moldescriptor)
            
        elif feature == "fp":
            fp = data.morgan_fp.numpy().flatten()
            X.append(fp)
            
    X = np.array(X)
    return X

def smiles_to_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"failed SMILES: {smiles}")

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

    array = list(fingerprint.ToBitString()) 
    array = [int(bit) for bit in array]
    
    tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0)
    
    return tensor

def create_molecule_data_quantum_chemistry_data(df):
    
    data_list = []

    idx_counter = 0
    
    for _, row in df.iterrows():
        name = row['Name']
        smiles = row['SMILES']
        smiles = add_hydrogens_to_smiles(smiles)
        
        morgan_fp_tensor = smiles_to_morgan_fingerprint(smiles, radius=2, n_bits=2048)
        
        tensors_to_check = [
            morgan_fp_tensor,
        ]

        if any(torch.isnan(t).any() or torch.isinf(t).any() for t in tensors_to_check):
            print(f"{name} contain NaN or Inf.")
            continue

        data = Data(
            idx=torch.tensor([idx_counter], dtype=torch.long),
            name=name,
            smiles=smiles,
            morgan_fp = morgan_fp_tensor,
        )

        data_list.append(data)
        idx_counter += 1
        
    return data_list

def add_predictions_to_df(df: pd.DataFrame, 
                          y_ECW: np.ndarray, 
                          y_Tm: np.ndarray, 
                          y_conductivity: np.ndarray) -> pd.DataFrame:


    n_rows = len(df)
    if not (len(y_ECW) == n_rows and len(y_Tm) == n_rows and len(y_conductivity) == n_rows):
        raise ValueError("len(df) != len(y_ECW) or len(df) != len(y_Tm) or len(df) != len(y_conductivity)")
    
    df_updated = df.copy()
    
    df_updated["ECW (V)"] = y_ECW
    df_updated["Tm (K)"] = y_Tm
    df_updated["conductivity (mS/cm)"] = y_conductivity
    
    return df_updated

def predict_property(df, output_file_path):
    data_list = create_molecule_data_quantum_chemistry_data(df)
    X = extract_features_targets(data_list)

    y_ECW = IL_ECW_xgb_model.predict(X)
    y_Tm = Tm_xgb_model.predict(X)
    model_conductivity_MLP.eval()
    y_conductivity = []

    for data in data_list:
        data = data.to(device)
        data.morgan_fp = data.morgan_fp.float() 
        out = model_conductivity_MLP(data)

        y_conductivity.append(out.detach().cpu().numpy())
    y_conductivity = np.concatenate(y_conductivity, axis=0)
    y_conductivity = y_conductivity.flatten() 

    df_result = add_predictions_to_df(df, y_ECW, y_Tm, y_conductivity)
    df_result.to_csv(output_file_path, index=None)  