from typing import List, Union, Optional

import torch
import pandas as pd
import torch.utils


class GenericFlightMecaDataset(torch.utils.data.Dataset):
    def __init__(
        self, df: pd.DataFrame, var_names: List[str], var_names_list: List[List[str]]
    ):
        self.x = torch.tensor(df.values, dtype=torch.float32)
        cols = list(df.columns)
        self.useful_cols = [cols.index(var) for var in var_names]
        for var_list in var_names_list:
            self.useful_cols.append([cols.index(var) for var in var_list])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return [self.x[index, col] for col in self.useful_cols]


class GenericFlightMecaDatasetFast(torch.utils.data.Dataset):
    """
    Faster than GenericFlightDataset but more memory intensive
    """

    def __init__(
        self, df: pd.DataFrame, var_names: List[str], var_names_list: List[List[str]]
    ):
        self.x = torch.tensor(df.values, dtype=torch.float32)
        cols = list(df.columns)
        self.useful_cols = [cols.index(var) for var in var_names]
        self.useful_cols_list = []
        for var_list in var_names_list:
            self.useful_cols.append([cols.index(var) for var in var_list])
        self.tensors = []
        for useful_col in self.useful_cols:
            self.tensors.append(self.x[:, useful_col])
        for useful_cols in self.useful_cols_list:
            self.tensors.append(self.x[:, useful_cols])
        self.nb_tensors = len(self.tensors)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return [self.tensors[i][index] for i in range(self.nb_tensors)]
