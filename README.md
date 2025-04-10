# ILGen

**Coupling AI and High-Throughput Computation for Ionic Liquids: From Molecular Generation to Application**

## Features

1. **Core and Skeleton Extraction**  
   Process cation and anion data to extract core structures and skeletons.  
2. **New Ion Generation**  
   Combine extracted cores and skeletons to create novel cations and anions.  
3. **Property Prediction**  
   Predict properties of newly generated ionic liquids using pre-trained machine learning models.

---

## Installation

To install ILGen, ensure that you have Python (3.7 or higher) and pip installed. Follow these steps:

### Clone the Repository

```bash
git clone https://github.com/WangGroupFDU/ILGen
cd ILGen/ILGen
pip install .
```
---

### Usage

ILGen provides three core functionalities, accessible via command-line subcommands. Below are the detailed instructions for each subcommand.

## 1. Generate Core and Skeleton Files

Extract core structures and skeletons from cation and anion Excel files. Input files must include columns labeled SMILES and Name.

```bash
ILGen generate_core_fragment -c <cation_excel_file> -a <anion_excel_file>
```

**Arguments:**

- `-c` or `--cation`: Path to the cation Excel file (required).  
- `-a` or `--anion`: Path to the anion Excel file (required).  

Example:

```bash
ILGen generate_core_fragment -c Cation.xlsx -a Anion.xlsx
```

## 2. Generate New Cations and Anions

Create new cations and anions by combining extracted cores and skeletons.

```bash
ILGen generate_new_cation_anion \
    --cation_core_excel <cation_core_file> \
    --anion_core_excel <anion_core_file> \
    --cation_backbone_excel <cation_backbone_file> \
    --anion_backbone_excel <anion_backbone_file>
```

**Arguments:**

- `--cation_core_excel`: Path to the cation core file (required).  
- `--anion_core_excel`: Path to the anion core file (required).  
- `--cation_backbone_excel`: Path to the cation backbone file (required).  
- `--anion_backbone_excel`: Path to the anion backbone file (required).  

Example:
```bash
ILGen generate_new_cation_anion \
    --cation_core_excel Cation_core.xlsx \
    --anion_core_excel Anion_core.xlsx \
    --cation_backbone_excel Cation_backbone.xlsx \
    --anion_backbone_excel Anion_backbone.xlsx
```

## 3. Predict and Generate Properties for Ionic Liquids

Predict properties of ionic liquids using the generated cation and anion CSV files.

```bash
ILGen generate_predict_new_IL \
    --new_cation_path <cation_csv> \
    --new_anion_path <anion_csv> \
    [--cation_limit <limit>] \
    [--anion_limit <limit>] \
    [--seed <seed>]
```

**Arguments:**

- `--new_cation_path`: Path to the new cation CSV file (required).  
- `--new_anion_path`: Path to the new anion CSV file (required).  
- `--cation_limit`: Maximum number of cations (default: 5000).  
- `--anion_limit`: Maximum number of anions (default: 300).  
- `--seed`: Random seed for reproducibility (default: 1).  

Example:

```bash
ILGen generate_predict_new_IL \
    --new_cation_path New_Cation.csv \
    --new_anion_path New_Anion.csv \
    --cation_limit 1000 \
    --anion_limit 100
```

## Example Workflow

```bash
cd ILGen/ILGen/exmaple
```

1. Generate Core and Skeleton Files:
```bash
ILGen generate_core_fragment -c Cation.xlsx -a Anion.xlsx
```
2. Create New Cations and Anions:
```bash
ILGen generate_new_cation_anion \
    --cation_core_excel Cation_core.xlsx \
    --anion_core_excel Anion_core.xlsx \
    --cation_backbone_excel Cation_backbone.xlsx \
    --anion_backbone_excel Anion_backbone.xlsx
```
3. Predict and Generate Properties:
```bash
ILGen generate_predict_new_IL \
    --new_cation_path New_Cation.csv \
    --new_anion_path New_Anion.csv \
    --cation_limit 1000 \
    --anion_limit 100
```

## Notes
	•	Input files must conform to the expected format, including required columns such as SMILES and Name.
	•	For optimal performance, ensure all dependencies are correctly installed.

---

# Data Sources for Machine Learning Models

```bash
cd ILGen/dataset
```

This repository includes datasets used for training machine learning models to predict properties of ionic liquids. The data covers a range of electrochemical and quantum chemical properties, which are described in detail below:

## Dataset Overview

### **1. Conductivity**
- **Number of records**: 549
- **File**: `filtered_ILThermo_conductivity_final.csv`
- **Description**: Experimental data measuring the ionic conductivity of ionic liquids.

### **2. Melting Point**
- **Number of records**: 1,668
- **File**: `experiment_data_IL_melting_point.csv`
- **Description**: Experimental data on the melting points of ionic liquids.

### **3. Redox Potential**
- **Number of records**:  1,578 
  - **Cations**: 768  
  - **Anions**: 166  
- **File**: `calculation_data_IL_ECW_IP-EA.csv`, `Cation_redox_potential.xlsx`, `Anion_redox_potential.xlsx`
- **Description**: Redox potential data for ionic liquids calculated using the IP-EA method.

### **4. Cation Quantum Chemistry Data**
- **Number of records**: 3,774  
- **File**: `HTQC_cation.xlsx`  
- **Description**: High-throughput quantum chemical calculations for cations.

### **5. Anion Quantum Chemistry Data**
- **Number of records**: 2,220  
- **File**: `HTQC_anion.xlsx`  
- **Description**: High-throughput quantum chemical calculations for anions.

## License

ILGen is licensed under the MIT License. See the LICENSE file for details.

