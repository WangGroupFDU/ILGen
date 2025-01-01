# ILGen

ILGen: An Ionic Liquid Generator Based on the BRICS Algorithm

## Features

1. **Core and Skeleton Extraction**: Process cation and anion data to extract core structures and skeletons.
2. **New Ion Generation**: Combine extracted cores and skeletons to create novel cations and anions.
3. **Property Prediction**: Predict properties of newly generated ionic liquids using pre-trained machine learning models.

---

## Installation

To install ILGen, ensure that you have Python (3.7 or higher) and pip installed. Follow these steps:

### Clone the Repository
git clone https://github.com/WangGroupFDU/ILGen.git
cd ILGen
pip install .

## Usage

ILGen provides three core functionalities, accessible via command-line subcommands. Below are the detailed instructions for each subcommand.

### 1. Generate Core and Skeleton Files

Extract core structures and skeletons from cation and anion Excel files. Input files must include columns labeled SMILES and Name.

ILGen generate_core_fragment -c <cation_excel_file> -a <anion_excel_file>

Arguments:
	•	-c or --cation: Path to the cation Excel file (required).
	•	-a or --anion: Path to the anion Excel file (required).
 
Example:
ILGen generate_core_fragment -c Cation.xlsx -a Anion.xlsx

### 2. Generate New Cations and Anions

Create new cations and anions by combining extracted cores and skeletons.

ILGen generate_new_cation_anion \
    --cation_core_excel <cation_core_file> \
    --anion_core_excel <anion_core_file> \
    --cation_backbone_excel <cation_backbone_file> \
    --anion_backbone_excel <anion_backbone_file>

Arguments:
	•	--cation_core_excel: Path to the cation core file (required).
	•	--anion_core_excel: Path to the anion core file (required).
	•	--cation_backbone_excel: Path to the cation backbone file (required).
	•	--anion_backbone_excel: Path to the anion backbone file (required).

Example:
ILGen generate_new_cation_anion \
    --cation_core_excel Cation_core.xlsx \
    --anion_core_excel Anion_core.xlsx \
    --cation_backbone_excel Cation_backbone.xlsx \
    --anion_backbone_excel Anion_backbone.xlsx

### 3. Predict and Generate Properties for Ionic Liquids

Predict properties of ionic liquids using the generated cation and anion CSV files.

ILGen generate_predict_new_IL \
    --new_cation_path <cation_csv> \
    --new_anion_path <anion_csv> \
    [--cation_limit <limit>] \
    [--anion_limit <limit>] \
    [--seed <seed>]

Arguments:
	•	--new_cation_path: Path to the new cation CSV file (required).
	•	--new_anion_path: Path to the new anion CSV file (required).
	•	--cation_limit: Maximum number of cations (default: 5000).
	•	--anion_limit: Maximum number of anions (default: 300).
	•	--seed: Random seed for reproducibility (default: 1).

Example:
ILGen generate_predict_new_IL \
    --new_cation_path New_Cation.csv \
    --new_anion_path New_Anion.csv \
    --cation_limit 1000 \
    --anion_limit 100

## Example Workflow

cd ILGen/ILGen/exmaple

### 1. Generate Core and Skeleton Files:

ILGen generate_core_fragment -c Cation.xlsx -a Anion.xlsx

### 2. Create New Cations and Anions:

ILGen generate_new_cation_anion \
    --cation_core_excel Cation_core.xlsx \
    --anion_core_excel Anion_core.xlsx \
    --cation_backbone_excel Cation_backbone.xlsx \
    --anion_backbone_excel Anion_backbone.xlsx

### 3. Predict and Generate Properties:
ILGen generate_predict_new_IL \
    --new_cation_path New_Cation.csv \
    --new_anion_path New_Anion.csv \
    --cation_limit 1000 \
    --anion_limit 100

## Notes
	•	Input files must conform to the expected format, including required columns such as SMILES and Name.
	•	For optimal performance, ensure all dependencies are correctly installed.

## License

ILGen is licensed under the MIT License. See the LICENSE file for details.
