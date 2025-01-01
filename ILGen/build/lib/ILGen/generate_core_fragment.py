import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
import os
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import MaxNLocator
from math import factorial
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import pairwise_distances
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def calculate_charge(smiles):
    mol = Chem.MolFromSmiles(smiles) 
    mol = Chem.AddHs(mol) 
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) 
    return charge

def get_filename_without_extension(xlsx_path):
    base_name = os.path.basename(xlsx_path)
    file_name_without_extension = os.path.splitext(base_name)[0]
    return file_name_without_extension

def normalization(mol):
    smi=Chem.MolToSmiles(mol)
    n_mol=Chem.MolFromSmiles(smi)
    return n_mol

def normalization_SMILES(excel_path, smiles_col_to_normalize):
    df = pd.read_excel(excel_path)
    file_name_without_extension = get_filename_without_extension(excel_path)
    
    if smiles_col_to_normalize in df.columns:
        invalid_rows = []
        for index, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row[smiles_col_to_normalize])
                if mol:  
                    n_mol = normalization(mol)
                    if n_mol: 
                        n_smi = Chem.MolToSmiles(n_mol)
                        df.at[index, smiles_col_to_normalize] = n_smi
                    else:
                        invalid_rows.append(index)
                else:
                    invalid_rows.append(index)
            except Exception as e:
                print(f"An error occurred for index {index}: {e}")
                invalid_rows.append(index)
        
        df.drop(invalid_rows, inplace=True)
        df.to_excel(f'{file_name_without_extension}.xlsx', index=False)
    else:
        print(f"The '{smiles_col_to_normalize}' column does not exist in the provided Excel file.")

def filter_star_elements(*lists):
    return [[item for item in lst if '[*]' in item] for lst in lists]

def decompose_ions_to_excel(excel_path, cation_cores, cation_backbones, anion_cores, anion_backbones):
    df = pd.read_excel(excel_path) 
    file_name_without_extension = get_filename_without_extension(excel_path)

    for index, row in df.iterrows():
        smiles = row['SMILES']  
        mol = Chem.MolFromSmiles(smiles)  
        mol = normalization(mol) 
        fragments = BRICS.BreakBRICSBonds(mol)
        fragments_smiles = Chem.MolToSmiles(fragments).split('.') 

        fragments_list = []
        for frag_smiles in fragments_smiles:
            frag_smiles_clean = re.sub(r'\[\d+\*\]', '[*]', frag_smiles)
            fragments_list.append(frag_smiles_clean)

        for frag_smiles in fragments_list:
            charge = calculate_charge(frag_smiles)  
            if charge > 0:
                cation_cores.append(frag_smiles)
            elif charge < 0:
                anion_cores.append(frag_smiles)
            else:
                if '+' in smiles:
                    cation_backbones.append(frag_smiles)
                elif '-' in smiles:
                    anion_backbones.append(frag_smiles)
                else:
                    cation_backbones.append(frag_smiles)

def decompose_ions_to_col(excel_path):
    df = pd.read_excel(excel_path)
    file_name_without_extension = get_filename_without_extension(excel_path)
    df['Core_SMILES'] = None
    df['Backbone_SMILES'] = None

    for index, row in df.iterrows():
        smiles = row['SMILES']  
        mol = Chem.MolFromSmiles(smiles)  
        mol = normalization(mol)  
        fragments = BRICS.BreakBRICSBonds(mol) 
        fragments_smiles = Chem.MolToSmiles(fragments).split('.')  

        core_smiles = []
        backbone_smiles = []

        for frag_smiles in fragments_smiles:
            frag_smiles_clean = re.sub(r'\[\d+\*\]', '[*]', frag_smiles) 
            charge = calculate_charge(frag_smiles_clean) 
            if charge > 0:
                core_smiles.append(frag_smiles_clean)
            elif charge < 0:
                core_smiles.append(frag_smiles_clean)
            else:
                if '+' in smiles:
                    backbone_smiles.append(frag_smiles_clean)
                elif '-' in smiles:
                    backbone_smiles.append(frag_smiles_clean)
                else:
                    backbone_smiles.append(frag_smiles_clean)

        df.at[index, 'Core_SMILES'] = '.'.join(core_smiles)
        df.at[index, 'Backbone_SMILES'] = '.'.join(backbone_smiles)

    df.to_excel(f"{file_name_without_extension}.xlsx", index=None)
    return df

def filter_list(input_list):
    return [item for item in input_list if item.count('[*]') <= 1]

def filter_molecule(xlsx_path):
    df = pd.read_excel(xlsx_path)
    file_name_without_extension = get_filename_without_extension(xlsx_path)

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

    filtered_molecule.to_excel(f'{file_name_without_extension}.xlsx', index=False)
    
    print(f"Skipped {file_name_without_extension}: ", skipped_molecule)

def count_virtual_sites(smiles):
    return smiles.count('[*]')

def filter_virtual_site_number(excel_file):
    df = pd.read_excel(excel_file)
    
    df['virtual_site_number'] = df['SMILES'].apply(count_virtual_sites)
    filtered_df = df[df['virtual_site_number'] <= 4]
    filtered_df.to_excel(excel_file, index=False)

def remove_duplicates_from_excel(excel_path, *columns):
    try:
        df = pd.read_excel(excel_path)
        file_name_without_extension = get_filename_without_extension(excel_path)

        for column in columns:
            if column not in df.columns:
                raise ValueError(f"'{column}' not exist")

        deduplicated_df = df.drop_duplicates(subset=columns, keep='first')

        deduplicated_df.to_excel(f'{file_name_without_extension}.xlsx', index=False)

    except FileNotFoundError:
        print(f"{excel_path} not found")
    except Exception as e:
        print(f"error: {e}")

def fill_empty_names_in_excel(excel_path):
    try:
        df = pd.read_excel(excel_path)
        file_name_without_extension = get_filename_without_extension(excel_path)

        if 'Name' not in df.columns:
            raise ValueError("'Name' col not exist")

        name_counter = 1 
        for i, row in df.iterrows():
            if pd.isnull(row['Name']):
                df.at[i, 'Name'] = f'{file_name_without_extension}name{name_counter}'
                name_counter += 1
        df.to_excel(f'{file_name_without_extension}.xlsx', index=False)

    except FileNotFoundError:
        print(f"{excel_path} not found")
    except Exception as e:
        print(f"error: {e}")

def count_excel_rows(excel_path):
    xls = pd.ExcelFile(excel_path)
    file_name_without_extension = get_filename_without_extension(excel_path)
    total_rows = 0
    for sheet_name in xls.sheet_names:
        sheet_data = pd.read_excel(excel_path, sheet_name=sheet_name)
        total_rows += len(sheet_data)
    print(f'Total data of {file_name_without_extension}: {total_rows}')
    return total_rows

def create_excel_from_lists(cation_core_list, cation_backbone_list, anion_core_list, anion_backbone_list):
    def generate_excel(data_list, file_name, id_prefix):
        df = pd.DataFrame({'SMILES': data_list})
        df = df.drop_duplicates().reset_index(drop=True)
        df['SerialNumber'] = df.index + 1
        df['Name'] = id_prefix + df['SerialNumber'].astype(str)
        df = df[['SerialNumber', 'Name', 'SMILES']]
        df.to_excel(file_name, index=False)
    
    generate_excel(cation_core_list, 'Cation_core.xlsx', 'c-c')
    generate_excel(cation_backbone_list, 'Cation_backbone.xlsx', 'c-b')
    generate_excel(anion_core_list, 'Anion_core.xlsx', 'a-c')
    generate_excel(anion_backbone_list, 'Anion_backbone.xlsx', 'a-b')


def generate_core_fragment(cation_excel_file_path, anion_excel_file_path):
    normalization_SMILES(cation_excel_file_path, smiles_col_to_normalize="SMILES")
    normalization_SMILES(anion_excel_file_path, smiles_col_to_normalize="SMILES")

    filter_molecule(cation_excel_file_path)
    filter_molecule(anion_excel_file_path)

    remove_duplicates_from_excel(cation_excel_file_path, 'SMILES')
    remove_duplicates_from_excel(anion_excel_file_path, 'SMILES')

    fill_empty_names_in_excel(cation_excel_file_path)
    fill_empty_names_in_excel(anion_excel_file_path)
    
    cation_cores = []
    cation_backbones = []
    anion_cores = []
    anion_backbones = []

    decompose_ions_to_excel(cation_excel_file_path, cation_cores, cation_backbones, anion_cores, anion_backbones)
    decompose_ions_to_excel(anion_excel_file_path, cation_cores, cation_backbones, anion_cores, anion_backbones)
    decompose_ions_to_col(cation_excel_file_path)
    decompose_ions_to_col(anion_excel_file_path)
    
    cation_cores = list(set(cation_cores))
    cation_backbones = list(set(cation_backbones))
    anion_cores = list(set(anion_cores))
    anion_backbones = list(set(anion_backbones))

    cation_cores, cation_backbones, anion_cores, anion_backbones = filter_star_elements(
        cation_cores, cation_backbones, anion_cores, anion_backbones
    )

    cation_backbones = filter_list(cation_backbones)
    anion_backbones = filter_list(anion_backbones)

    create_excel_from_lists(cation_cores, cation_backbones, anion_cores, anion_backbones)

    filter_molecule('Cation_core.xlsx')
    filter_molecule('Cation_backbone.xlsx')
    filter_molecule('Anion_core.xlsx')
    filter_molecule('Anion_backbone.xlsx')

    filter_virtual_site_number('Cation_core.xlsx')
    filter_virtual_site_number('Cation_backbone.xlsx')
    filter_virtual_site_number('Anion_core.xlsx')
    filter_virtual_site_number('Anion_backbone.xlsx')

    Cation_core_data_count = count_excel_rows('Cation_core.xlsx')
    Cation_backbone_data_count = count_excel_rows('Cation_backbone.xlsx')
    Anion_core_data_count = count_excel_rows('Anion_core.xlsx')
    Anion_backbone_data_count = count_excel_rows('Anion_backbone.xlsx')

    print(
        f"Processing completed. Generated files:\n"
        f"- 'Cation_core.xlsx' ({Cation_core_data_count} entries)\n"
        f"- 'Cation_backbone.xlsx' ({Cation_backbone_data_count} entries)\n"
        f"- 'Anion_core.xlsx' ({Anion_core_data_count} entries)\n"
        f"- 'Anion_backbone.xlsx' ({Anion_backbone_data_count} entries)"
    )

# test
# generate_core_fragment("Cation.xlsx", "Anion.xlsx")