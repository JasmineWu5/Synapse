import copy
import logging

import awkward as ak
import numpy as np
from torch.utils.data import Dataset, IterableDataset

from core.tools import apply_selection, build_new_variables, extract_fields_from_expr
from core.fileio import read_files
from core.config import DataConfig

_logger = logging.getLogger("SynapseLogger")

def prepare_data(data: ak.Array, data_cfg: DataConfig, dataset_type: str) -> dict:
    """
    Prepare the data according to the configuration.
    """
    # Apply selection and build new variables
    if data_cfg.selection:
        data = apply_selection(data, data_cfg.selection, data_cfg.new_variables)
    else:
        data = build_new_variables(data, data_cfg.new_variables)
    # TODO: implement padding for variable length arrays
    # TODO: check if the data is empty after selection
    # TODO: check if the data only have one class in classification tasks
    # TODO: check if the data have entries with more/less than one active label in classification tasks
    # Check if all active variables are valid
    # activate_variables: inputs, label_variables, weights
    # spectators don't need to be checked, since they will not be fed to the model
    for var in data_cfg.active_variables:
        if np.any(np.isnan(data[var])):
            data[var] = np.nan_to_num(data[var], posinf=np.inf, neginf=-np.inf)
            _logger.warning(f"NaN values found in variable '{var}', replaced with zeros.")
        if np.any(np.isinf(data[var])):
            raise RuntimeError(f"Infinite values found in variable '{var}'.") # FIXME: should be handled more gracefully?
    # Convert to numpy arrays, otherwise we need to define a complex collate function
    prepared_data = {}
    for k, vs in data_cfg.inputs.items():
        prepared_data[k] = ak.to_numpy(ak.concatenate([data[v][:, np.newaxis] for v in vs], axis=1)).astype(np.float32)
    for k in data_cfg.final_label_variables:
        prepared_data[k] = ak.to_numpy(data[k])
    if data_cfg.final_weight_variable in data.fields:
        prepared_data[data_cfg.final_weight_variable] = ak.to_numpy(data[data_cfg.final_weight_variable])
    if dataset_type == 'test':
        for k in data_cfg.spectators:
            prepared_data[k] = ak.to_numpy(data[k])

    return prepared_data

# class HybridDataset(IterableDataset):
#     def __init__(self, file_list):
#         self.file_list = file_list
#
#     def __iter__(self):
#         for file in self.file_list:
#             chunk_data = load_chunk(file)
#
#             map_dataset = MapStyleDataset(chunk_data)
#             yield from map_dataset

# TODO: decide whether to support lazy loading or not
class MapStyleDataset(Dataset):
    def __init__(self, file_list: list[str], dataset_type: str, data_cfg: DataConfig):
        self.dataset_type = dataset_type
        self.data_cfg = data_cfg
        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid dataset type: {dataset_type}. Must be one of ['train', 'val', 'test'].")
        keys = data_cfg.active_variables
        if dataset_type == 'test':
            keys.extend(data_cfg.spectators)
        if data_cfg.new_variables:
            for expr in data_cfg.new_variables.values():
                keys.extend(extract_fields_from_expr(expr))
        load_range = tuple(data_cfg.get(f'{dataset_type}_load_range', (0.0,1.0)))
        normalized_range = data_cfg.get(f'{dataset_type}_normalized_range', True)
        raw_data, _ = read_files(file_list, list(set(keys)), load_range, normalized_range, merge=True, tree_name=data_cfg.get('tree_name'))
        self.data = prepare_data(raw_data, data_cfg, dataset_type)

        lengths = [len(arr) for arr in self.data.values()]
        if len(set(lengths)) != 1:
            raise ValueError("All feature arrays must have the same length")
        self.num_samples = lengths[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Don't use np.ndarray.copy() here, as it will return a view of the original array in some cases
        # TODO: copy.deepcopy() is not efficient, consider using a more efficient method if needed
        x = {k: copy.deepcopy(self.data[k][idx]) for k in self.data_cfg.inputs.keys()}
        y = {k: copy.deepcopy(self.data[k][idx]) for k in self.data_cfg.final_label_variables}
        w = {self.data_cfg.final_weight_variable:  copy.deepcopy(self.data[self.data_cfg.final_weight_variable][idx])}
        s = {}
        # TODO: spectators will be converted to tensors, but not necessary, we need to improve this
        if self.dataset_type == 'test':
            s = {k:  copy.deepcopy(self.data[k][idx]) for k in self.data_cfg.spectators}
        return x, y, w, s