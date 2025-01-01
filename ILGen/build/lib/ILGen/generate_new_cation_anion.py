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
import operator

IPythonConsole.ipython_useSVG = True
logger = RDLogger.logger()
logger.setLevel(RDLogger.ERROR)
logger.setLevel(RDLogger.CRITICAL)

bond_list = [Chem.rdchem.BondType.UNSPECIFIED,
             Chem.rdchem.BondType.SINGLE, 
             Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, 
             Chem.rdchem.BondType.QUADRUPLE, 
             Chem.rdchem.BondType.QUINTUPLE,
             Chem.rdchem.BondType.HEXTUPLE, 
             Chem.rdchem.BondType.ONEANDAHALF, 
             Chem.rdchem.BondType.TWOANDAHALF,
             Chem.rdchem.BondType.THREEANDAHALF, 
             Chem.rdchem.BondType.FOURANDAHALF, 
             Chem.rdchem.BondType.FIVEANDAHALF,
             Chem.rdchem.BondType.AROMATIC, 
             Chem.rdchem.BondType.IONIC, 
             Chem.rdchem.BondType.HYDROGEN,
             Chem.rdchem.BondType.THREECENTER,
             Chem.rdchem.BondType.DATIVEONE,
             Chem.rdchem.BondType.DATIVE,
             Chem.rdchem.BondType.DATIVEL, 
             Chem.rdchem.BondType.DATIVER,
             Chem.rdchem.BondType.OTHER,
             Chem.rdchem.BondType.ZERO]

def create_atom_object(smiles, ID):
    mol = Chem.MolFromSmiles(smiles)
    
    atom_object_list = []
    
    atoms_formal_charged_dict = {}
    
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx() 
        atom_symbol = atom.GetSymbol()
        atom_formal_charge = atom.GetFormalCharge()  
        if atom_symbol == '*': 
            atom_name = f"*{atom_idx}_{ID}"
        else:
            atom_name = f"{atom_symbol}{atom_idx}_{ID}" 

        atom_object_list.append(atom_name)

        if atom_formal_charge != 0:
            atoms_formal_charged_dict[atom_name] = atom_formal_charge
            
    return atom_object_list, atoms_formal_charged_dict

def all_bonds_info_and_order(bond_list, smiles, ID):
    mol = Chem.MolFromSmiles(smiles)

    all_bonds_info = []
    bonds_info_with_order = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        begin_atom_symbol = mol.GetAtomWithIdx(begin_idx).GetSymbol()
        end_atom_symbol = mol.GetAtomWithIdx(end_idx).GetSymbol()
        begin_atom_name = f"{begin_atom_symbol}{begin_idx}_{ID}"
        end_atom_name = f"{end_atom_symbol}{end_idx}_{ID}"
        bond_order = bond_list.index(bond.GetBondType())

        if begin_atom_symbol == '*':
            begin_atom_name = f"*{begin_idx}_{ID}"
        if end_atom_symbol == '*':
            end_atom_name = f"*{end_idx}_{ID}"

        bonds_info_with_order.append(((begin_atom_name, end_atom_name), bond_order))

        all_bonds_info.append((begin_atom_name, end_atom_name))
    
    dummy_atom_bonds_info = [bond for bond in all_bonds_info if '*' in bond[0] or '*' in bond[1]]
    
    return all_bonds_info, bonds_info_with_order, dummy_atom_bonds_info

def count_virtual_sites(smiles):
    return smiles.count('[*]')

def fragments_data_preprocessing(bond_list, excel_file_path):
    df = pd.read_excel(excel_file_path)

    df['atom_object_list'], df['atoms_formal_charged_dict'] = zip(*df.apply(
        lambda row: create_atom_object(row['SMILES'], row['Name']), axis=1))

    df['all_bonds_info'], df['bonds_info_with_order'], df['dummy_atom_bonds_info'] = zip(*df.apply(
        lambda row: all_bonds_info_and_order(bond_list, row['SMILES'], row['Name']), axis=1))
    
    df['virtual_site_number'] = df['SMILES'].apply(count_virtual_sites)
    df.to_excel(excel_file_path, index=False)
    print(f"Bonds information and atom object have been updated and exported to {excel_file_path}")
    
    return df

def remove_rows_with_virtual_site_number_greater_than_one(df):

    if 'virtual_site_number' not in df.columns:
        print("DataFrame中不存在'virtual_site_number'列。")
        return df
    
    filtered_df = df[df['virtual_site_number'] <= 1]
    
    return filtered_df

def remove_starred_atoms(atom_list):
    filtered_list = [atom for atom in atom_list if "*" not in atom]
    return filtered_list

def create_element_node_list(node_list):
    element_node_list = []

    pattern = re.compile(r"([A-Z][a-z]*)(?=\d*_?[a-z]*-\w*\d*)")

    for node in node_list:
        if node.startswith("*"): 
            element_node_list.append("*")
        else:
            match = pattern.match(node)
            if match:
                element_name = match.group(1)
                element_node_list.append(element_name)
            else:
                element_node_list.append(node)
    
    return element_node_list

def create_adjacency_matrix(ion_atom_list, ion_connection_and_order_list):
    adjacency_matrix = np.zeros((len(ion_atom_list), len(ion_atom_list)))

    node_index = {node: idx for idx, node in enumerate(ion_atom_list)}
    
    for bond, order in ion_connection_and_order_list:
        node1, node2 = bond
        idx1, idx2 = node_index[node1], node_index[node2]
        adjacency_matrix[idx1][idx2] = order
        adjacency_matrix[idx2][idx1] = order 
    return adjacency_matrix

def create_index_formal_charge_dict(node_list, atoms_formal_charged_dict):
    index_formal_charge_dict = {}
    for index, node in enumerate(node_list):
        if node in atoms_formal_charged_dict:
            index_formal_charge_dict[index] = atoms_formal_charged_dict[node]

    return index_formal_charge_dict

def graph2mol(atoms, ad_martrix, index_formal_charge_dict):
    new_mol = Chem.RWMol()
    atom_index = []

    for atom_number in range(len(atoms)):
        atom = Chem.Atom(atoms[atom_number]) 
        molecular_index = new_mol.AddAtom(atom) 
        atom_index.append(molecular_index)  
    
    for index_x, row_vector in enumerate(ad_martrix):
        for index_y, bond in enumerate(row_vector):
            bond = int(bond)
            if index_y <= index_x:
                continue
            if bond == 0:
                continue
            else:
                new_mol.AddBond(atom_index[index_x],
                                atom_index[index_y], 
                                bond_list[bond]) 
  
    for atom_index, formal_charge in index_formal_charge_dict.items():
        atom = new_mol.GetAtomWithIdx(atom_index)
        neighbors = atom.GetNeighbors()
        for neighbor in neighbors:
            if neighbor.GetSymbol() == 'H':
                editable_mol.RemoveAtom(neighbor.GetIdx())

        atom.SetFormalCharge(formal_charge)
    try:
        Chem.SanitizeMol(new_mol)
        
        new_mol = new_mol.GetMol()
        
        new_mol = Chem.RemoveHs(new_mol)
        
        # 返回新创建的分子对象
        return new_mol
    
    except (AtomValenceException, KekulizeException) as e:
        return None
    except Exception as e:
        return None

def has_unique_second_elements(pair_list):
    second_elements = {pair[1] for pair in pair_list}
    return len(second_elements) == len(pair_list)

def ion_reorganization(df_core, df_backbone):
    data = []
    counter_success = 0
    counter = 0 
    for index, row in df_core.iterrows():
        n = row['virtual_site_number']
        core_id_list = [] 
        core_id = row['Name']
        core_id_list.append(core_id)
        core_all_bonds_info = row['all_bonds_info'] 
        core_bonds_info_with_order = row['bonds_info_with_order'] 
        core_atom_object_list = row['atom_object_list'] 
        core_atoms_formal_charged_dict = row['atoms_formal_charged_dict'] 
        core_dummy_atom_bonds_info = row['dummy_atom_bonds_info'] 
        core_smiles = row['SMILES'] 

        core_connection_site_list = [] 
        for i, bond in enumerate(core_dummy_atom_bonds_info):
            if '*' in bond[0]:
                core_connection_site_list.append(bond[1])
            else:
                core_connection_site_list.append(bond[0])
    
        if not isinstance(n, int):
            raise ValueError(f"The value in virtual_site_number must be an integer, got {n} instead.")
        backbone_combinations = combinations(df_backbone.index, n)
        for combo_indices in backbone_combinations:
            combo_backbone_id_list = []
            combo_backbone_all_bonds_info = [] 
            combo_backbone_bonds_info_with_order = [] 
            combo_backbone_atom_object_list = [] 
            combo_backbone_dummy_atom_bonds_info = [] 
    
            for idx in combo_indices:
                backbone_row = df_backbone.loc[idx]
                combo_backbone_id_list.append(backbone_row['Name'])
                backbone_all_bonds_info = backbone_row['all_bonds_info']
                combo_backbone_all_bonds_info.extend(backbone_all_bonds_info)
                
                backbone_bonds_info_with_order = backbone_row['bonds_info_with_order']
                combo_backbone_bonds_info_with_order.extend(backbone_bonds_info_with_order)
                
                backbone_atom_object_list = backbone_row['atom_object_list']
                combo_backbone_atom_object_list.extend(backbone_atom_object_list)
                
                backbone_dummy_atom_bonds_info = backbone_row['dummy_atom_bonds_info']
                combo_backbone_dummy_atom_bonds_info.extend(backbone_dummy_atom_bonds_info)
    
            ion_atom_list =  core_atom_object_list + combo_backbone_atom_object_list 
            inner_connection_list = core_all_bonds_info + combo_backbone_all_bonds_info
            inner_connection_and_order_list = core_bonds_info_with_order + combo_backbone_bonds_info_with_order
            ion_ID_list =  core_id_list + combo_backbone_id_list
    
            backbone_connection_site_list = []
            for i, bond in enumerate(combo_backbone_dummy_atom_bonds_info):
                if '*' in bond[0]:
                    backbone_connection_site_list.append(bond[1])
                else:
                    backbone_connection_site_list.append(bond[0])
    
            ion_ID = '+'.join(ion_ID_list)
            
            ion_atom_node_list = remove_starred_atoms(ion_atom_list)
            ion_atom_list = create_element_node_list(ion_atom_node_list)

            inner_connection_and_order_list = [
                bond_info for bond_info in inner_connection_and_order_list
                if '*' not in bond_info[0][0] and '*' not in bond_info[0][1]
            ]

            inner_connection_list = [
                bond for bond in inner_connection_list if '*' not in bond[0] and '*' not in bond[1]
            ]
    
            all_combinations = list(product(core_connection_site_list, backbone_connection_site_list))
            all_outer_connection_list = [list(zip(core_connection_site_list, p)) for p in product(backbone_connection_site_list, repeat=len(core_connection_site_list))]
            filtered_outer_connection_list = [combo for combo in all_outer_connection_list if has_unique_second_elements(combo)] 
            if filtered_outer_connection_list:
                outer_connection_list = random.choice(filtered_outer_connection_list)

                outer_connection_and_order_list = [(connection, 1) for connection in outer_connection_list]

                ion_connection_list = outer_connection_list + inner_connection_list

                ion_connection_and_order_list = outer_connection_and_order_list + inner_connection_and_order_list # 重组后的离子连接关系与键级
    

                adjacency_matrix = create_adjacency_matrix(ion_atom_node_list, ion_connection_and_order_list)
                ion_index_formal_charge_dict = create_index_formal_charge_dict(ion_atom_node_list, core_atoms_formal_charged_dict)
                new_mol = graph2mol(ion_atom_list, adjacency_matrix, ion_index_formal_charge_dict)
                counter += 1
                if new_mol is None:
                    continue
                else:
                    new_smiles = Chem.MolToSmiles(new_mol)

                    if "." not in new_smiles:
                        SAscore = sascorer.calculateScore(new_mol)
                        data.append([counter_success + 1, ion_ID, new_smiles, SAscore])
                        counter_success += 1
            else:
                print(f"skip: core_id: {core_id},combo_backbone_id_list:{combo_backbone_id_list}")
                continue

    print(f"Attempted to generate {counter} ions, successfully generated {counter_success} ions, with a validity rate of {counter_success/counter:.2f}")
    df = pd.DataFrame(data, columns=['SerialNumber', 'Name', 'SMILES', 'SAscore'])
    return df

def get_filename_without_extension(file_path):
    base_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(base_name)[0]
    return file_name_without_extension

def filter_molecule(df):
    period_5_elements = ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"]
    period_6_elements = ["Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"]
    period_7_elements = ["Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

    metals_and_after_period_4 = [
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"
    ] + period_5_elements + period_6_elements + period_7_elements

    skipped_molecule = set()

    def contains_forbidden_atoms(smiles, forbidden_atoms):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in forbidden_atoms:
                    return True
        return False
    
    filtered_molecule = df[~df['SMILES'].apply(contains_forbidden_atoms, args=(metals_and_after_period_4,))]

    skipped_molecule.update(df.loc[df.index.difference(filtered_molecule.index), 'Name'])

    # print(f"Skipped: ", skipped_molecule)

    return filtered_molecule

def generate_new_cation_anion(cation_core_excel, anion_core_excel, cation_backbone_excel, anion_backbone_excel):
    df_cation_core = fragments_data_preprocessing(bond_list, cation_core_excel)
    df_anion_core = fragments_data_preprocessing(bond_list, anion_core_excel)
    df_cation_backbone = fragments_data_preprocessing(bond_list, cation_backbone_excel)
    df_anion_backbone = fragments_data_preprocessing(bond_list, anion_backbone_excel)

    df_cation_backbone = remove_rows_with_virtual_site_number_greater_than_one(df_cation_backbone)
    df_anion_backbone = remove_rows_with_virtual_site_number_greater_than_one(df_anion_backbone)

    start_time = time.time()
    df_new_cation = ion_reorganization(df_cation_core, df_cation_backbone)
    end_time = time.time()
    print(f"Generation of cations took a total of {end_time - start_time} seconds.")

    start_time = time.time()
    df_new_anion = ion_reorganization(df_anion_core, df_anion_backbone)
    end_time = time.time()
    print(f"Generation of anions took a total of {end_time - start_time} seconds.")

    filtered_df_new_cation = filter_molecule(df_new_cation)
    filtered_df_new_anion = filter_molecule(df_new_anion)

    filtered_df_new_cation.to_csv('New_Cation.csv', index=False)
    filtered_df_new_anion.to_csv('New_Anion.csv', index=False)

# test
# generate_new_cation_anion("Cation_core.xlsx", "Anion_core.xlsx", "Cation_backbone.xlsx", "Anion_backbone.xlsx")