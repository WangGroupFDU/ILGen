# -*- coding: utf-8 -*-
"""
cli.py

Command-line interface for the Ionic Liquid Generator (ILGen).
This script provides three core functionalities:
    1) generate_core_fragment: Processes cation and anion data to extract core structures and skeletons.
    2) generate_new_cation_anion: Combines cation and anion cores with skeletons to create new ions.
    3) generate_predict_new_IL: Predicts properties for newly generated ionic liquids (ILs).

Each functionality is implemented as a distinct subcommand, accessible via the command line.
Users must provide specific input files and parameters, as described below.
"""

import argparse
import sys

# Importing the core functions from the ILGen package
# These functions handle the main computational tasks associated with ionic liquid generation.
from ILGen.generate_core_fragment import generate_core_fragment
from ILGen.generate_new_cation_anion import generate_new_cation_anion
from ILGen.generate_predict_new_IL import generate_predict_new_IL


def main():
    """
    Main entry point for the ILGen command-line interface.
    This function parses command-line arguments, identifies the requested subcommand, 
    and delegates execution to the corresponding function.
    """

    # Initializing the top-level argument parser
    parser = argparse.ArgumentParser(
        description=(
            "ILGen: A comprehensive tool for generating ionic liquids. "
            "Supports operations including core/skeleton extraction, ion recombination, and IL property prediction."
        )
    )

    # Adding subparsers for distinct functionalities
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # -------------------------------------------------------------------
    # Subcommand 1: generate_core_fragment
    # Description: Processes cation and anion data to generate cores and skeletons.
    # -------------------------------------------------------------------
    parser_fragment = subparsers.add_parser(
        "generate_core_fragment",
        help=(
            "Extracts core structures and skeletons from cation and anion Excel files. "
            "Both files must include columns labeled 'SMILES' and 'Name'."
        )
    )
    parser_fragment.add_argument(
        "-c", "--cation",
        required=True,
        help="Path to the cation Excel file containing 'SMILES' and 'Name' columns."
    )
    parser_fragment.add_argument(
        "-a", "--anion",
        required=True,
        help="Path to the anion Excel file containing 'SMILES' and 'Name' columns."
    )

    # -------------------------------------------------------------------
    # Subcommand 2: generate_new_cation_anion
    # Description: Combines cores and skeletons to create new ions.
    # -------------------------------------------------------------------
    parser_new = subparsers.add_parser(
        "generate_new_cation_anion",
        help=(
            "Generates new cations and anions by combining extracted cores and skeletons. "
            "Input files must be generated using the 'generate_core_fragment' command."
        )
    )
    parser_new.add_argument(
        "--cation_core_excel",
        required=True,
        help="Path to the Excel file containing cation cores."
    )
    parser_new.add_argument(
        "--anion_core_excel",
        required=True,
        help="Path to the Excel file containing anion cores."
    )
    parser_new.add_argument(
        "--cation_backbone_excel",
        required=True,
        help="Path to the Excel file containing cation skeletons."
    )
    parser_new.add_argument(
        "--anion_backbone_excel",
        required=True,
        help="Path to the Excel file containing anion skeletons."
    )

    # -------------------------------------------------------------------
    # Subcommand 3: generate_predict_new_IL
    # Description: Predicts properties for newly generated ionic liquids.
    # -------------------------------------------------------------------
    parser_predict = subparsers.add_parser(
        "generate_predict_new_IL",
        help=(
            "Predicts properties of newly generated ionic liquids. "
            "Input files must be generated using the 'generate_new_cation_anion' command."
        )
    )
    parser_predict.add_argument(
        "--new_cation_path",
        required=True,
        help="Path to the CSV file containing new cations."
    )
    parser_predict.add_argument(
        "--new_anion_path",
        required=True,
        help="Path to the CSV file containing new anions."
    )
    parser_predict.add_argument(
        "--cation_limit",
        type=int,
        default=5000,
        help="Maximum number of cations to include in predictions (default: 5000)."
    )
    parser_predict.add_argument(
        "--anion_limit",
        type=int,
        default=300,
        help="Maximum number of anions to include in predictions (default: 300)."
    )
    parser_predict.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility (default: 1)."
    )

    # Parsing the command-line arguments
    args = parser.parse_args()

    # If no subcommand is provided, display the help message and exit
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Delegating execution to the appropriate function based on the subcommand
    if args.command == "generate_core_fragment":
        # Execute the core fragment generation workflow
        generate_core_fragment(
            cation_excel_file_path=args.cation,
            anion_excel_file_path=args.anion
        )

    elif args.command == "generate_new_cation_anion":
        # Execute the new ion generation workflow
        generate_new_cation_anion(
            cation_core_excel=args.cation_core_excel,
            anion_core_excel=args.anion_core_excel,
            cation_backbone_excel=args.cation_backbone_excel,
            anion_backbone_excel=args.anion_backbone_excel
        )

    elif args.command == "generate_predict_new_IL":
        # Execute the IL property prediction workflow
        generate_predict_new_IL(
            new_cation_path=args.new_cation_path,
            new_anion_path=args.new_anion_path,
            cation_limit=args.cation_limit,
            anion_limit=args.anion_limit,
            seed=args.seed
        )

    else:
        # If an unrecognized command is provided, display the help message and exit
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    # Entry point for the script
    main()