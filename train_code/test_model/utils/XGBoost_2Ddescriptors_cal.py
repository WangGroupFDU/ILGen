import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdPartialCharges, Fragments
from rdkit.Chem.GraphDescriptors import BalabanJ, BertzCT
from rdkit.Chem.EState.EState_VSA import (EState_VSA1, EState_VSA2, EState_VSA3, EState_VSA4, EState_VSA5,
                                          EState_VSA6, EState_VSA7, EState_VSA8, EState_VSA9, EState_VSA10)
from rdkit.Chem.EState import EState_VSA

def calculate_all_molecular_descriptors(smiles):
    """
    功能说明：
    该函数将原有的多个分子描述符计算函数整合为一个函数。函数输入为一个SMILES字符串形式的分子结构，
    输出为一个张量(tensor)，形状为[1, NumMoldescriptors]，其中NumMoldescriptors为所有分子描述符的数量之和。
    
    描述符类型包括：
    1. 基于图论的分子描述符（BalabanJ、BertzCT）
    2. EState_VSA描述符（EState_VSA1 ~ EState_VSA10）
    3. 各类传统分子描述符（ExactMolWt、FractionCSP3、HallKierAlpha...等）
    4. Gasteiger偏电荷相关描述符（MaxAbsPartialCharge, MaxPartialCharge, MinAbsPartialCharge, MinPartialCharge）
    5. 融合了环数量、TPSA、SMR_VSA、SlogP_VSA、VSA_EState等描符
    6. 各种片段（Fragment）相关描述符（fr_开头）
    7. 基于SMARTS模式搜索的一系列结构特征计数(fr_nitro_arom_nonortho, fr_nitroso, ...)

    最终将所有描述符合并为一个Series并转化为Tensor返回。

    参数：
    smiles (str): 分子SMILES字符串
    
    返回：
    torch.Tensor: 大小为 [1, NumMoldescriptors] 的张量（浮点类型）
    """
    # 从SMILES生成分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 若分子解析失败，则返回一个空Tensor
        return torch.zeros(1, 0)

    #==========================================================
    # 以下为原代码中各函数的描述符计算内容的合并
    #==========================================================
    
    #------------------------------
    # 来自calculate_ion_properties_1的描述符
    # 包括BalabanJ、BertzCT和EState_VSA1~10
    #------------------------------
    desc_1 = {
        'BalabanJ': BalabanJ(mol),
        'BertzCT': BertzCT(mol),
        'EState_VSA1': EState_VSA1(mol),
        'EState_VSA2': EState_VSA2(mol),
        'EState_VSA3': EState_VSA3(mol),
        'EState_VSA4': EState_VSA4(mol),
        'EState_VSA5': EState_VSA5(mol),
        'EState_VSA6': EState_VSA6(mol),
        'EState_VSA7': EState_VSA7(mol),
        'EState_VSA8': EState_VSA8(mol),
        'EState_VSA9': EState_VSA9(mol),
        'EState_VSA10': EState_VSA10(mol)
    }

    #------------------------------
    # 来自calculate_ion_properties_2的描述符
    # 包括ExactMolecularWeight、FractionCSP3、HallKierAlpha、Ipc、Kappa1~3、LabuteASA等
    # 以及Gasteiger偏电荷相关的Max/Min（绝对值）PartialCharge
    #------------------------------
    properties_2 = {
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

    # 为Gasteiger电荷计算添加氢原子
    mol_with_h = Chem.AddHs(mol)
    rdPartialCharges.ComputeGasteigerCharges(mol_with_h)
    charges = [mol_with_h.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol_with_h.GetNumAtoms())]
    properties_2['MaxAbsPartialCharge'] = max(charges, key=abs)
    properties_2['MaxPartialCharge'] = max(charges)
    properties_2['MinAbsPartialCharge'] = min(charges, key=abs)
    properties_2['MinPartialCharge'] = min(charges)

    #------------------------------
    # 来自calculate_ion_properties_3的描述符
    # 分子基本性质：MolLogP、MolMR、MolWt、NHOHCount、NOCount、各种环计数、受氢计数、供氢计数等
    #------------------------------
    desc_3 = {
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

    #------------------------------
    # 来自calculate_ion_properties_4的描述符
    # 包括NumRotatableBonds、NumSaturatedCarbocycles、NumSaturatedHeterocycles、NumSaturatedRings、NumValenceElectrons、PEOE_VSA系列
    #------------------------------
    # 注：无需Embed和MMFF优化，因为PEOE_VSA的计算不依赖3D坐标
    mol_h = Chem.AddHs(mol)  # 部分描述符可能更依赖氢原子存在
    desc_4 = {
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol_h),
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(mol_h),
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(mol_h),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol_h),
        'NumValenceElectrons': Descriptors.NumValenceElectrons(mol_h)
    }

    peoe_vsa = rdMolDescriptors.PEOE_VSA_(mol_h)
    for i, value in enumerate(peoe_vsa, start=1):
        desc_4[f"PEOE_VSA{i}"] = value

    #------------------------------
    # 来自calculate_ion_properties_5的描述符
    # 包括RingCount、TPSA、SMR_VSA系列、SlogP_VSA系列、VSA_EState系列
    #------------------------------
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    # SMR_VSA
    smr_vsa = rdMolDescriptors.SMR_VSA_(mol)
    # SlogP_VSA
    slogp_vsa = rdMolDescriptors.SlogP_VSA_(mol)
    # VSA_EState
    vsa_estate = EState_VSA.EState_VSA_(mol)

    desc_5 = {
        "RingCount": ring_count,
        "TPSA": tpsa
    }
    for i, val in enumerate(smr_vsa, start=1):
        desc_5[f"SMR_VSA{i}"] = val
    for i, val in enumerate(slogp_vsa, start=1):
        desc_5[f"SlogP_VSA{i}"] = val
    for i, val in enumerate(vsa_estate, start=1):
        desc_5[f"VSA_EState{i}"] = val

    #------------------------------
    # 来自calculate_ion_properties_6的描述符 (Fragments)
    #------------------------------
    desc_6 = {
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
        'fr_amide': Fragments.fr_amide(mol)
    }

    #------------------------------
    # 来自calculate_ion_properties_7的描述符 (Fragments)
    #------------------------------
    desc_7 = {
        'fr_amidine': Fragments.fr_amidine(mol),
        'fr_aniline': Fragments.fr_aniline(mol),
        'fr_aryl_methyl': Fragments.fr_aryl_methyl(mol),
        'fr_azide': Fragments.fr_azide(mol),
        'fr_azo': Fragments.fr_azo(mol),
        'fr_barbitur': Fragments.fr_barbitur(mol),
        'fr_benzene': Fragments.fr_benzene(mol),
        'fr_benzodiazepine': Fragments.fr_benzodiazepine(mol),
        'fr_bicyclic': Fragments.fr_bicyclic(mol),
        'fr_diazo': Fragments.fr_diazo(mol),
        'fr_dihydropyridine': Fragments.fr_dihydropyridine(mol),
        'fr_epoxide': Fragments.fr_epoxide(mol),
        'fr_ester': Fragments.fr_ester(mol),
        'fr_ether': Fragments.fr_ether(mol),
        'fr_furan': Fragments.fr_furan(mol),
        'fr_guanido': Fragments.fr_guanido(mol),
        'fr_halogen': Fragments.fr_halogen(mol),
        'fr_hdrzine': Fragments.fr_hdrzine(mol),
        'fr_hdrzone': Fragments.fr_hdrzone(mol),
        'fr_imidazole': Fragments.fr_imidazole(mol),
        'fr_imide': Fragments.fr_imide(mol),
        'fr_isocyan': Fragments.fr_isocyan(mol),
        'fr_isothiocyan': Fragments.fr_isothiocyan(mol),
        'fr_ketone': Fragments.fr_ketone(mol),
        'fr_ketone_Topliss': Fragments.fr_ketone_Topliss(mol),
        'fr_lactam': Fragments.fr_lactam(mol),
        'fr_lactone': Fragments.fr_lactone(mol),
        'fr_methoxy': Fragments.fr_methoxy(mol),
        'fr_morpholine': Fragments.fr_morpholine(mol),
        'fr_nitrile': Fragments.fr_nitrile(mol),
        'fr_nitro': Fragments.fr_nitro(mol),
        'fr_nitro_arom': Fragments.fr_nitro_arom(mol)
    }

    #------------------------------
    # 来自calculate_ion_properties_using_smarts的描述符
    # 这里采用SMARTS匹配计数特定功能团特征
    #------------------------------
    smarts_patterns = {
        'fr_nitro_arom_nonortho': '[$([N+]([O-])=O),$([N+](=O)[O-])][!#1]:[c]:[c]',
        'fr_nitroso': '[N](=O)[O-]',
        'fr_oxazole': 'n1ccoc1',
        'fr_oxime': '[NX2H][C]=[O]',
        'fr_para_hydroxylation': '',  # 空模式，计数为0
        'fr_phenol': 'c1ccccc1O',
        'fr_phenol_noOrthoHbond': '',  # 空模式，计数为0
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
        'fr_urea': 'NC(=O)N'
    }

    desc_smarts = {}
    for feature, smarts in smarts_patterns.items():
        if smarts:
            patt = Chem.MolFromSmarts(smarts)
            desc_smarts[feature] = len(mol.GetSubstructMatches(patt))
        else:
            desc_smarts[feature] = 0

    #==========================================================
    # 将所有描述符字典合并为一个pandas.Series
    #==========================================================
    all_desc = {}
    all_desc.update(desc_1)
    all_desc.update(properties_2)
    all_desc.update(desc_3)
    all_desc.update(desc_4)
    all_desc.update(desc_5)
    all_desc.update(desc_6)
    all_desc.update(desc_7)
    all_desc.update(desc_smarts)

    desc_series = pd.Series(all_desc)

    # 转换为numpy数组
    desc_array = desc_series.values.astype(float)
    # 转换为torch张量
    desc_tensor = torch.tensor(desc_array, dtype=torch.float32).unsqueeze(0)  # [1, NumMoldescriptors]

    return desc_tensor