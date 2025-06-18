# a script to convert the CAF flat ntuple to a tabular format ROOT file
import argparse
import glob
import os
from pathlib import Path

import awkward as ak
import yaml

from core.fileio import read_files, write_file
from core.tools import build_new_variables

def load_config(config_path) -> dict:
    """
    Load the configuration file and extract the branches to be used

    Args:
        config_path (str): path to the configuration file
    Returns:
        dict: configuration dictionary with branches
    """
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)

    branches = set()
    for obj_name in cfg.get('object_names', []):
        for var in cfg.get('object_variables', []):
            branches.add(f'{obj_name}_{var}')

    for var in cfg.get('event_variables', []):
        branches.add(var)

    cfg['branches'] = list(branches)

    return cfg

def convert(input_data: ak.Array, cfg: dict):
    """
    Convert the input flat ntuple to a tabular format
    """
    output_data = {}
    for feat in cfg.get('object_variables', []):
        arrays = [input_data[f"{obj_name}_{feat}"] for obj_name in cfg.get('object_names', [])]
        output_data[feat] = ak.concatenate([arr[:, None] for arr in arrays], axis=1)

    for feat in cfg.get('event_variables', []):
        output_data[feat] = input_data[feat]

    output_data = ak.zip(output_data, depth_limit=1)

    output_data = build_new_variables(output_data, cfg.get('new_variables'))

    return output_data

def split_folds(input_data: ak.Array, cfg: dict) -> list[ak.Array]:
    """
    Split the data into folds
    """
    remainders = input_data.eventNumber % cfg.get('k_folds', 1) # FIXME: hardcoded to eventNumber, need to improve flexibility
    folds = [input_data[remainders == r] for r in range(cfg.get('k_folds', 1))]

    return folds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ROOT file to tabular format")
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')

    args = parser.parse_args()

    config = load_config(args.config)

    in_file_paths = []
    for file_path in config.get('in_file_paths', []):
        in_file_paths.extend(glob.glob(file_path))

    data_in, file_names_in = read_files(file_paths=in_file_paths,
                                        keys=config['branches'],
                                        merge=config['merge_input'],
                                        tree_name=config.get('in_tree_name'))

    if config.get('merge_input'):
        data_out = convert(data_in, config)
        if config.get('k_folds', 1) > 1:
            data_out_folds = split_folds(data_out, config)
            for i, fold in enumerate(data_out_folds):
                file_path_out = os.path.join(config['output_dir'], "merged" ,f"fold_{i}.root")
                write_file(file_path_out, fold, tree_name=config.get('out_tree_name', 'tree'))
        else:
            file_path_out = os.path.join(config['output_dir'], "merged" ,f"merged_total.root")
            write_file(file_path_out, data_out, tree_name=config.get('out_tree_name', 'tree'))
    else:
        data_out = []
        for data in data_in:
            data_out.append(convert(data, config))
        if config.get('k_folds', 1) > 1:
            for i, data in enumerate(data_out):
                data_folds = split_folds(data, config)
                for j, fold in enumerate(data_folds):
                    sub_dir_name = Path(file_names_in[i]).stem
                    file_path_out = os.path.join(config['output_dir'], sub_dir_name ,f"fold_{j}.root")
                    write_file(file_path_out, fold, tree_name=config.get('out_tree_name', 'tree'))
        else:
            for i, data in enumerate(data_out):
                sub_dir_name = Path(file_names_in[i]).stem
                file_path_out = os.path.join(config['output_dir'], sub_dir_name ,f"merged_total.root")
                write_file(file_path_out, data, tree_name=config.get('out_tree_name', 'tree'))





