from logging import getLogger
from typing import Union
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.exceptions import NotFittedError

from .flight_meca_net_3d_eq import FlightMecaNet3DEq
from .datasets import GenericFlightMecaDatasetFast


class PhysicalModel:
    def __init__(
        self,
        log_dir: Path,
        saving_dir: Path,
        saving_name: str,
        NetClass: type[pl.LightningModule] = FlightMecaNet3DEq,
        net_params: Union[dict, None] = None,
        DatasetClass: type[torch.utils.data.Dataset] = GenericFlightMecaDatasetFast,
        dataset_params: Union[dict, None] = None,
        trainer_params: Union[dict, None] = None,
        batch_size: int = 128,
        num_loader_workers: int = 8,
    ):
        self.logger = getLogger("PhysicalModel")

        self.log_dir = log_dir
        self.saving_dir = saving_dir
        self.saving_name = saving_name
        self.NetClass = NetClass
        self.net_params = dict() if net_params is None else deepcopy(net_params)
        self.DatasetClass = DatasetClass
        self.dataset_params = (
            dict() if dataset_params is None else deepcopy(dataset_params)
        )
        self.trainer_params = (
            dict() if trainer_params is None else deepcopy(trainer_params)
        )
        self.batch_size = batch_size
        self.num_loader_workers = num_loader_workers

        self.trainer = pl.Trainer(**trainer_params)
        self._is_fitted = False

        try:
            self.load(self.saving_dir / (self.saving_name + ".ckpt"))
            self.logger.info("Model correctly loaded")
        except FileNotFoundError:
            self.logger.warning(
                f"Saving file not founded for the model loading, please train the model. Saving file path : {self.saving_dir / (self.saving_name + '.ckpt')}"
            )
            self.model = self.NetClass(**self.net_params)

    def fit(self, data):
        train_dataset = self.DatasetClass(data.train, **self.dataset_params)
        val_dataset = self.DatasetClass(data.test, **self.dataset_params)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_loader_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_loader_workers,
        )
        self.trainer.fit(
            self.model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        self._is_fitted = True

    def predict(self, df: pd.DataFrame, concat_predict_and_data: bool = True):
        if not self._is_fitted:
            raise NotFittedError(
                "The model is not fitted yet. Use 'fit' or 'load' methods."
            )
        dataset = self.DatasetClass(df, **self.dataset_params)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_loader_workers,
        )

        predictions = self.trainer.predict(self.model, dataloaders=loader)
        result_df = self._df_from_predictions(df, predictions)
        if concat_predict_and_data:
            return pd.concat([df, result_df], axis="columns", join="inner")
        else:
            return result_df

    def load(self, ckpt: Path):
        self.model = self.NetClass.load_from_checkpoint(ckpt, **self.net_params)
        self._is_fitted = True

    def _df_from_predictions(self, original_df, predictions):
        df = pd.DataFrame()
        batch_dfs = list()
        for batch_index in range(len(predictions)):
            batch_df = pd.DataFrame()
            for i, var_name in enumerate(self.NetClass.get_output_names()):
                batch_df[var_name] = predictions[batch_index][i]
            batch_dfs.append(batch_df)
        df = pd.concat(batch_dfs, axis="index")
        df.index = original_df.index
        # df.set_index("time", inplace=True)
        return df
