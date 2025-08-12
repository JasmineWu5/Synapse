# a script to convert the CAF flat ntuple to the format suitable for Synapse
import argparse
import glob
import math
import os
from pathlib import Path

import awkward as ak
import yaml

from synapse.core.fileio import read_files, write_file
from synapse.core.tools import build_new_variables

def valid_config(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path

def load_config(config_path) -> dict:
    """
    Load the configuration file and extract the branches to be used

    Args:
        config_path (str): path to the configuration file
    Returns:
        dict: configuration dictionary with branches
    """
    print(f"Loading configuration from: {config_path}")
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
    Convert the input flat ntuple
    """
    output_data = {}

    for feat in cfg.get('object_variables', []):
        arrays = [input_data[f"{obj_name}_{feat}"] for obj_name in cfg.get('object_names', [])]
        output_data[feat] = ak.concatenate([arr[:, None] for arr in arrays], axis=1)

    for feat in cfg.get('event_variables', []):
        output_data[feat] = input_data[feat]

    output_data = ak.zip(output_data, depth_limit=1)

    output_data = build_new_variables(output_data, cfg.get('new_variables'))

    # outlier replacement
    for field, value_pair in cfg.get('outlier_replacements', {}).items():
        if field in output_data.fields:
            if math.isnan(value_pair[0]):
                output_data[field] = ak.fill_none(ak.nan_to_none(output_data[field]), value_pair[1], axis=None)
            else:
                output_data[field] = ak.where(output_data[field] == value_pair[0], value_pair[1], output_data[field])
        else:
            raise ValueError(f"Outlier replacement: \nField '{field}' not found in output data. Available fields: {output_data.fields}")

    return output_data

def split_folds(input_data: ak.Array, cfg: dict, fold_splitting_var: str) -> list[ak.Array]:
    """
    Split the data into folds
    """
    remainders = input_data[fold_splitting_var] % cfg.get('k_folds', 1)
    folds = [input_data[remainders == r] for r in range(cfg.get('k_folds', 1))]

    return folds

def main():
    parser = argparse.ArgumentParser(description="Convert ROOT file")
    parser.add_argument('-c','--config', type=str, required=True, help='Configuration file path')

    args = parser.parse_args()

    print("Starting hhml CAF ntuple conversion process...")

    config = load_config(valid_config(args.config))

    in_file_paths = []
    for file_path in config.get('in_file_paths', []):
        in_file_paths.extend(glob.glob(file_path))

    print("Converting...")

    data_in, file_names_in = read_files(file_paths=in_file_paths,
                                        keys=config['branches'],
                                        merge=config['merge_input'],
                                        tree_name=config.get('in_tree_name'))

    if config.get('merge_input'):
        data_out = convert(data_in, config)
        if config.get('k_folds', 1) > 1:
            data_out_folds = split_folds(data_out, config, config.get('fold_splitting_var', 'eventNumber'))
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
                data_folds = split_folds(data, config, config.get('fold_splitting_var', 'eventNumber'))
                for j, fold in enumerate(data_folds):
                    sub_dir_name = Path(file_names_in[i]).stem
                    file_path_out = os.path.join(config['output_dir'], sub_dir_name ,f"fold_{j}.root")
                    write_file(file_path_out, fold, tree_name=config.get('out_tree_name', 'tree'))
        else:
            for i, data in enumerate(data_out):
                sub_dir_name = Path(file_names_in[i]).stem
                file_path_out = os.path.join(config['output_dir'], sub_dir_name ,f"merged_total.root")
                write_file(file_path_out, data, tree_name=config.get('out_tree_name', 'tree'))

    print("Finished.")
if __name__ == "__main__":
    main()




