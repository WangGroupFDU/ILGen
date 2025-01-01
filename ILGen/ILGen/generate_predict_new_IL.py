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
from rdkit.Contrib.SA_Score import sascorer
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

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.GraphDescriptors import BalabanJ, BertzCT
from rdkit.Chem.EState.EState_VSA import (EState_VSA1, EState_VSA2, EState_VSA3, EState_VSA4, EState_VSA5,
                                          EState_VSA6, EState_VSA7, EState_VSA8, EState_VSA9, EState_VSA10)
from rdkit.Chem import rdMolDescriptors
import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import csv as csv
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV   #Performing grid search
from scipy.stats import skew
from collections import OrderedDict
from sklearn.inspection import permutation_importance
import shap
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import joblib
from rdkit.Chem import Descriptors, rdMolDescriptors, rdPartialCharges
from rdkit.Chem.EState import EState_VSA
from rdkit.Chem import Fragments

def calculate_ion_properties_1(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    descriptors = {}
    descriptors['BalabanJ'] = BalabanJ(molecule)
    descriptors['BertzCT'] = BertzCT(molecule)
    descriptors['EState_VSA1'] = EState_VSA1(molecule)
    descriptors['EState_VSA2'] = EState_VSA2(molecule)
    descriptors['EState_VSA3'] = EState_VSA3(molecule)
    descriptors['EState_VSA4'] = EState_VSA4(molecule)
    descriptors['EState_VSA5'] = EState_VSA5(molecule)
    descriptors['EState_VSA6'] = EState_VSA6(molecule)
    descriptors['EState_VSA7'] = EState_VSA7(molecule)
    descriptors['EState_VSA8'] = EState_VSA8(molecule)
    descriptors['EState_VSA9'] = EState_VSA9(molecule)
    descriptors['EState_VSA10'] = EState_VSA10(molecule)
    return pd.Series(descriptors)

def calculate_ion_properties_2(smiles):
    mol = Chem.MolFromSmiles(smiles)
    properties = {
        'ExactMolecularWeight': Descriptors.ExactMolWt(mol),
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
        'HallKierAlpha': rdMolDescriptors.CalcHallKierAlpha(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'HeavyAtomMolecularWeight': Descriptors.HeavyAtomMolWt(mol),
        'Ipc': Descriptors.Ipc(mol),
        'Kappa1': Descriptors.Kappa1(mol),
        'Kappa2': Descriptors.Kappa2(mol),
        'Kappa3': Descriptors.Kappa3(mol),
        'LabuteASA': rdMolDescriptors.CalcLabuteASA(mol),
        'MaxAbsEStateIndex': Descriptors.MaxAbsEStateIndex(mol),
        'MaxEStateIndex': Descriptors.MaxEStateIndex(mol),
        'MinAbsEStateIndex': Descriptors.MinAbsEStateIndex(mol)
    }
    mol_with_h = Chem.AddHs(mol)
    rdPartialCharges.ComputeGasteigerCharges(mol_with_h)
    charges = [mol_with_h.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol_with_h.GetNumAtoms())]
    properties['MaxAbsPartialCharge'] = max(charges, key=abs)
    properties['MaxPartialCharge'] = max(charges)
    properties['MinAbsPartialCharge'] = min(charges, key=abs)
    properties['MinPartialCharge'] = min(charges)

    return pd.Series(properties)

def calculate_ion_properties_3(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descriptors = {
        'MolLogP': Descriptors.MolLogP(mol),
        'MolMR': Descriptors.MolMR(mol),
        'MolWt': Descriptors.MolWt(mol),
        'NHOHCount': Descriptors.NHOHCount(mol),
        'NOCount': Descriptors.NOCount(mol),
        'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles(mol),
        'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles(mol),
        'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'NumRadicalElectrons': Descriptors.NumRadicalElectrons(mol)
    }

    return pd.Series(descriptors)

def calculate_ion_properties_4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    properties = {
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol),
        'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
    }

    peoe_vsa = rdMolDescriptors.PEOE_VSA_(mol)
    for i, value in enumerate(peoe_vsa, start=1):
        properties[f"PEOE_VSA{i}"] = value

    return pd.Series(properties)



def calculate_ion_properties_5(smiles):
    molecule = Chem.MolFromSmiles(smiles)

    ring_count = rdMolDescriptors.CalcNumRings(molecule)
    tpsa = rdMolDescriptors.CalcTPSA(molecule)

    smr_vsa = rdMolDescriptors.SMR_VSA_(molecule) 
    
    # Calculate SlogP_VSA descriptors
    slogp_vsa = rdMolDescriptors.SlogP_VSA_(molecule) 
  
    vsa_estate = EState_VSA.EState_VSA_(molecule)

    descriptors = {
        "RingCount": ring_count,
        "TPSA": tpsa,
        **{f"SMR_VSA{i}": smr_vsa[i-1] for i in range(1, len(smr_vsa)+1)},
        **{f"SlogP_VSA{i}": slogp_vsa[i-1] for i in range(1, len(slogp_vsa)+1)},
        **{f"VSA_EState{i}": vsa_estate[i-1] for i in range(1, len(vsa_estate)+1)}
    }
    
    return pd.Series(descriptors)

def calculate_ion_properties_6(smiles):
    mol = Chem.MolFromSmiles(smiles)

    fragments = {
        'fr_Al_COO': Fragments.fr_Al_COO(mol),
        'fr_Al_OH': Fragments.fr_Al_OH(mol),
        'fr_Al_OH_noTert': Fragments.fr_Al_OH_noTert(mol),
        'fr_ArN': Fragments.fr_ArN(mol),
        'fr_Ar_COO': Fragments.fr_Ar_COO(mol),
        'fr_Ar_N': Fragments.fr_Ar_N(mol),
        'fr_Ar_NH': Fragments.fr_Ar_NH(mol),
        'fr_Ar_OH': Fragments.fr_Ar_OH(mol),
        'fr_COO': Fragments.fr_COO(mol),
        'fr_COO2': Fragments.fr_COO2(mol),
        'fr_C_O': Fragments.fr_C_O(mol),
        'fr_C_O_noCOO': Fragments.fr_C_O_noCOO(mol),
        'fr_C_S': Fragments.fr_C_S(mol),
        'fr_HOCCN': Fragments.fr_HOCCN(mol),
        'fr_Imine': Fragments.fr_Imine(mol),
        'fr_NH0': Fragments.fr_NH0(mol),
        'fr_NH1': Fragments.fr_NH1(mol),
        'fr_NH2': Fragments.fr_NH2(mol),
        'fr_N_O': Fragments.fr_N_O(mol),
        'fr_Ndealkylation1': Fragments.fr_Ndealkylation1(mol),
        'fr_Ndealkylation2': Fragments.fr_Ndealkylation2(mol),
        'fr_Nhpyrrole': Fragments.fr_Nhpyrrole(mol),
        'fr_SH': Fragments.fr_SH(mol),
        'fr_aldehyde': Fragments.fr_aldehyde(mol),
        'fr_alkyl_carbamate': Fragments.fr_alkyl_carbamate(mol),
        'fr_alkyl_halide': Fragments.fr_alkyl_halide(mol),
        'fr_allylic_oxid': Fragments.fr_allylic_oxid(mol),
        'fr_amide': Fragments.fr_amide(mol),
    }

    return pd.Series(fragments)

def calculate_ion_properties_7(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    properties = {
        'fr_amidine': Fragments.fr_amidine(molecule),
        'fr_aniline': Fragments.fr_aniline(molecule),
        'fr_aryl_methyl': Fragments.fr_aryl_methyl(molecule),
        'fr_azide': Fragments.fr_azide(molecule),
        'fr_azo': Fragments.fr_azo(molecule),
        'fr_barbitur': Fragments.fr_barbitur(molecule),
        'fr_benzene': Fragments.fr_benzene(molecule),
        'fr_benzodiazepine': Fragments.fr_benzodiazepine(molecule),
        'fr_bicyclic': Fragments.fr_bicyclic(molecule),
        'fr_diazo': Fragments.fr_diazo(molecule),
        'fr_dihydropyridine': Fragments.fr_dihydropyridine(molecule),
        'fr_epoxide': Fragments.fr_epoxide(molecule),
        'fr_ester': Fragments.fr_ester(molecule),
        'fr_ether': Fragments.fr_ether(molecule),
        'fr_furan': Fragments.fr_furan(molecule),
        'fr_guanido': Fragments.fr_guanido(molecule),
        'fr_halogen': Fragments.fr_halogen(molecule),
        'fr_hdrzine': Fragments.fr_hdrzine(molecule),
        'fr_hdrzone': Fragments.fr_hdrzone(molecule),
        'fr_imidazole': Fragments.fr_imidazole(molecule),
        'fr_imide': Fragments.fr_imide(molecule),
        'fr_isocyan': Fragments.fr_isocyan(molecule),
        'fr_isothiocyan': Fragments.fr_isothiocyan(molecule),
        'fr_ketone': Fragments.fr_ketone(molecule),
        'fr_ketone_Topliss': Fragments.fr_ketone_Topliss(molecule),
        'fr_lactam': Fragments.fr_lactam(molecule),
        'fr_lactone': Fragments.fr_lactone(molecule),
        'fr_methoxy': Fragments.fr_methoxy(molecule),
        'fr_morpholine': Fragments.fr_morpholine(molecule),
        'fr_nitrile': Fragments.fr_nitrile(molecule),
        'fr_nitro': Fragments.fr_nitro(molecule),
        'fr_nitro_arom': Fragments.fr_nitro_arom(molecule),
    }

    return pd.Series(properties)

def calculate_ion_properties_using_smarts(smiles):
    mol = Chem.MolFromSmiles(smiles)

    smarts_patterns = {
        'fr_nitro_arom_nonortho': '[$([N+]([O-])=O),$([N+](=O)[O-])][!#1]:[c]:[c]',
        'fr_nitroso': '[N](=O)[O-]',
        'fr_oxazole': 'n1ccoc1',
        'fr_oxime': '[NX2H][C]=[O]',
        'fr_para_hydroxylation': '', 
        'fr_phenol': 'c1ccccc1O',
        'fr_phenol_noOrthoHbond': '',  
        'fr_phos_acid': 'P(=O)(O)(O)',
        'fr_phos_ester': 'P(=O)(O)[O-]',
        'fr_piperdine': 'C1CCNCC1',
        'fr_piperzine': 'C1CNCCN1',
        'fr_priamide': 'C(=O)N',
        'fr_prisulfonamd': 'S(=O)(=O)(N)',
        'fr_pyridine': 'n',
        'fr_quatN': '[N+](~*)(~*)(~*)(~*)',  
        'fr_sulfide': '[#16]',
        'fr_sulfonamd': 'S(=O)(=O)N',
        'fr_sulfone': 'S(=O)(=O)',
        'fr_term_acetylene': 'C#C',
        'fr_tetrazole': 'n1nnnc1',
        'fr_thiazole': 's1cc[nH]1',
        'fr_thiocyan': 'N=C=S',
        'fr_thiophene': 's1cccc1',
        'fr_unbrch_alkane': 'C(C)C',  
        'fr_urea': 'NC(=O)N',
    }

    features = {}
    for feature, smarts in smarts_patterns.items():
        if smarts: 
            pattern = Chem.MolFromSmarts(smarts)
            count = len(mol.GetSubstructMatches(pattern))
            features[feature] = count
        else:
            features[feature] = 0

    return pd.Series(features)

total_func_name = [
    "calculate_ion_properties_1",
    "calculate_ion_properties_2",
    "calculate_ion_properties_3",
    "calculate_ion_properties_4",
    "calculate_ion_properties_5",
    "calculate_ion_properties_6",
    "calculate_ion_properties_7",
    "calculate_ion_properties_using_smarts"
]

def cal_2D_descriptor(df, isILdata = True):
    if isILdata == True:
        total_cation_properties = pd.DataFrame()
        total_anion_properties = pd.DataFrame()

        for func_name in total_func_name:
            func = globals().get(func_name)
            if func:
                cation_properties = df['Cation_SMILES'].apply(func)
                anion_properties = df['Anion_SMILES'].apply(func)
                
                total_cation_properties = pd.concat([total_cation_properties, cation_properties], axis=1)
                total_anion_properties = pd.concat([total_anion_properties, anion_properties], axis=1)
            else:
                print(f"Function {func_name} not found.")
        
        total_cation_properties.columns = ['Cation_' + col for col in total_cation_properties.columns]
        total_anion_properties.columns = ['Anion_' + col for col in total_anion_properties.columns]
    
        df = pd.concat([df, total_cation_properties, total_anion_properties], axis=1)
        df = classify_smiles(df, "Cation_SMILES")
        return df

    if isILdata == False:
        total_ion_properties = pd.DataFrame()

        for func_name in total_func_name:
            func = globals().get(func_name)
            if func:
                properties = df['SMILES'].apply(func)
                total_ion_properties = pd.concat([total_ion_properties, properties], axis=1)
            else:
                print(f"Function {func_name} not found.")
    
        df = pd.concat([df, total_ion_properties], axis=1)
        return df
    
def classify_smiles(df, *col_names):
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
        df.loc[:, f'{col_name}_type'] = df[col_name].apply(match_type)

        type_counts = df[f'{col_name}_type'].value_counts()
        print(f"Type counts for column '{col_name}':")
        print(type_counts)
        print() 

        unknown_smiles = df[df[f'{col_name}_type'] == 'unknown'][col_name]
        if not unknown_smiles.empty:
            print(f"Unknown SMILES for column '{col_name}':")
            print(unknown_smiles)
            print()
        else:
            print(f"No unknown SMILES found in column '{col_name}'.\n")
        
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

def load_model_with_joblib_and_get_features(filename):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, filename)
    model = joblib.load(model_path)
    feature_names = model.get_booster().feature_names
    return model, feature_names



def combine_ion2IL(cation_df, anion_df, cation_limit=5000, anion_limit=300, seed=1):

    if 'Name' not in cation_df.columns or 'SMILES' not in cation_df.columns:
        raise ValueError("cation_df must contain both 'Name' and 'SMILES' columns.")
    cation_df = cation_df[['Name', 'SMILES']]
    
    # 保证 anion_df 有 'Name' 和 'SMILES' 列
    if 'Name' not in anion_df.columns or 'SMILES' not in anion_df.columns:
        raise ValueError("anion_df must contain both 'Name' and 'SMILES' columns.")
    anion_df = anion_df[['Name', 'SMILES']]

    cation_df = classify_smiles(cation_df, "SMILES")

    if len(cation_df) > cation_limit:
        cation_df = stratified_sampling(cation_df, col="SMILES_type", sample_size=cation_limit)
 
    if len(anion_df) > anion_limit:
        anion_df = anion_df.sample(n=anion_limit, random_state=seed).reset_index(drop=True)

    combinations = []

    cation_df_caldescriptor = cal_2D_descriptor(cation_df, isILdata = False)
    anion_df_caldescriptor = cal_2D_descriptor(anion_df, isILdata = False)

    for _, cation_row in cation_df_caldescriptor.iterrows():
        for _, anion_row in anion_df_caldescriptor.iterrows():
            combined_row = {
                "Name": f"{cation_row['Name']}.{anion_row['Name']}",
                "SMILES": f"{cation_row['SMILES']}.{anion_row['SMILES']}",
                "Anion_Name": anion_row['Name'],
                "Cation_Name": cation_row['Name'],
                "Anion_SMILES": anion_row['SMILES'],
                "Cation_SMILES": cation_row['SMILES'],
                "Cation_SMILES_type": cation_row['SMILES_type'],
            }

            for descriptor in cation_df_caldescriptor.columns:
                if descriptor not in ["Name", "SMILES", "SMILES_type"]:
                    combined_row[f"Cation_{descriptor}"] = cation_row[descriptor]

            for descriptor in anion_df_caldescriptor.columns:
                if descriptor not in ["Name", "SMILES"]:
                    combined_row[f"Anion_{descriptor}"] = anion_row[descriptor]

            combinations.append(combined_row)

    combined_df = pd.DataFrame(combinations)

    return combined_df

def predict_with_xgboost(df, model, features, result_column_name='prediction_conduvtivity(S/m)'):

    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        print("Missing feature columns:", missing_features)
        raise ValueError(f"The input DataFrame is missing the required feature columns: {missing_features}")
    X = df[features]
    predictions = model.predict(X)
    df_with_predictions = df.copy()
    df_with_predictions[result_column_name] = predictions

    return df_with_predictions

def generate_predict_new_IL(new_cation_path, new_anion_path, cation_limit=5000, anion_limit=300, seed=1):
    if new_cation_path.lower().endswith('.csv'):
        cation_df = pd.read_csv(new_cation_path)
    elif new_cation_path.lower().endswith(('.xls', '.xlsx')):
        cation_df = pd.read_excel(new_cation_path) 
    else:
        raise ValueError("Unsupported cation file type. Only CSV or Excel files are supported.")
    
    if new_anion_path.lower().endswith('.csv'):
        anion_df = pd.read_csv(new_anion_path) 
    elif new_anion_path.lower().endswith(('.xls', '.xlsx')):
        anion_df = pd.read_excel(new_anion_path) 
    else:
        raise ValueError("Unsupported anion file type. Only CSV or Excel files are supported.")
    
    IL_df = combine_ion2IL(cation_df, anion_df, cation_limit, anion_limit, seed)

    Tm_xgb_model, Tm_feature_names = load_model_with_joblib_and_get_features('xgboost_model/Tm_xgb_model.joblib')
    conductivity_xgb_model, conductivity_feature_names = load_model_with_joblib_and_get_features('xgboost_model/conductivity_xgb_model.joblib')
    ECW_xgb_model, ECW_feature_names = load_model_with_joblib_and_get_features('xgboost_model/ECW_xgb_model.joblib')

    predict_df = predict_with_xgboost(IL_df, Tm_xgb_model, Tm_feature_names, result_column_name='Tm(K)')
    predict_df = predict_with_xgboost(predict_df, conductivity_xgb_model, conductivity_feature_names, result_column_name='Conductivity(S/m)')
    predict_df = predict_with_xgboost(predict_df, ECW_xgb_model, ECW_feature_names, result_column_name='ECW(V)')
    total_count = len(predict_df)
    predict_df.to_csv("generate_IL.csv", index=False)

    print(f"The ionic liquid file 'generate_IL.csv' has been created in the current directory! A total of {total_count} ionic liquids were generated.")

# test
# generate_predict_new_IL("New_Cation.csv", "New_Anion.csv", cation_limit=5000, anion_limit=300, seed=1)