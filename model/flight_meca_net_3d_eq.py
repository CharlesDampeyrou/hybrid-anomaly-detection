import torch
import torch.nn as nn
import pytorch_lightning as pl

from .blocks import SimpleRegressor
from .flight_equations import (
    x_eq_aero,
    y_eq_aero,
    z_eq_aero,
    x_eq_aircraft,
    y_eq_aircraft,
    z_eq_aircraft,
    stengel_x_eq,
    stengel_y_eq,
    stengel_z_eq,
)


class FlightMecaNet3DEq(pl.LightningModule):
    def __init__(
        self,
        cx_param_dim,
        cy_param_dim,
        cz_param_dim,
        thrust_param_dim,
        regressor_layers: int = 3,
        regressor_layer_dim: int = 32,
        net_coef_dict: dict = None,
        lr=1e-3,
        equation_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        if net_coef_dict is None:
            net_coef_dict = {}
        if equation_params is None:
            equation_params = {}
        self.equation_params = equation_params
        cx_net_coef = net_coef_dict.get("cx_net_coef", 1.0)
        cy_net_coef = net_coef_dict.get("cy_net_coef", 1.0)
        cz_net_coef = net_coef_dict.get("cz_net_coef", 1.0)
        thrust_net_coef = net_coef_dict.get("thrust_net_coef", 1.0)
        self.cx_net = SimpleRegressor(
            input_dim=cx_param_dim,
            nb_hidden_layers=regressor_layers,
            layers_dim=regressor_layer_dim,
            output_multiplier=cx_net_coef,
            last_activation="LeakyReLU",
        )
        self.cy_net = SimpleRegressor(
            input_dim=cy_param_dim,
            nb_hidden_layers=regressor_layers,
            layers_dim=regressor_layer_dim,
            output_multiplier=cy_net_coef,
        )
        self.cz_net = SimpleRegressor(
            input_dim=cz_param_dim,
            nb_hidden_layers=regressor_layers,
            layers_dim=regressor_layer_dim,
            output_multiplier=cz_net_coef,
        )
        self.thrust_net = SimpleRegressor(
            input_dim=thrust_param_dim,
            nb_hidden_layers=regressor_layers,
            layers_dim=regressor_layer_dim,
            output_multiplier=thrust_net_coef,
            last_activation="LeakyReLU",
        )
        self.xi = torch.tensor(
            0.0, requires_grad=False
        )  # angle between the x-axis of the body frame and thrust direction
        self.x_eq = stengel_x_eq  # TODO: mettre possibilité de changer les eq
        self.y_eq = stengel_y_eq
        self.z_eq = stengel_z_eq
        self.loss_func = nn.HuberLoss()

    def forward(self, batch):
        (
            m,
            jx,
            jy,
            jz,
            alpha,
            beta,
            pression,
            temp,
            v,
            cx_inputs,
            cy_inputs,
            cz_inputs,
            thrust_inputs,
        ) = batch
        cx = self.cx_net(cx_inputs)
        cy = self.cy_net(cy_inputs)
        cz = self.cz_net(cz_inputs)
        thrust = self.thrust_net(thrust_inputs)
        eq_inputs = (
            m,
            jx,
            jy,
            jz,
            alpha,
            beta,
            self.xi,
            pression,
            temp,
            v,
            cx,
            cy,
            cz,
            thrust,
        )
        x_residue = self.x_eq(*eq_inputs, **self.equation_params)
        y_residue = self.y_eq(*eq_inputs, **self.equation_params)
        z_residue = self.z_eq(*eq_inputs, **self.equation_params)
        return (
            x_residue,
            y_residue,
            z_residue,
            cx,
            cy,
            cz,
            thrust,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        (
            x_residue,
            y_residue,
            z_residue,
            *args,
        ) = self.forward(batch)
        residue = torch.cat(
            (
                x_residue,
                y_residue,
                z_residue,
            )
        )
        loss = self.loss_func(residue, torch.zeros_like(residue))
        self.log("train loss", loss)
        self.log("train mean absolute x_residue", x_residue.abs().mean())
        self.log("train mean absolute y_residue", y_residue.abs().mean())
        self.log("train mean absolute z_residue", z_residue.abs().mean())
        return loss

    def validation_step(self, batch, batch_idx):
        (
            x_residue,
            y_residue,
            z_residue,
            *args,
        ) = self.forward(batch)
        residue = torch.cat(
            (
                x_residue,
                y_residue,
                z_residue,
            )
        )
        loss = self.loss_func(residue, torch.zeros_like(residue))
        self.log("val loss", loss)
        self.log("val mean absolute x_residue", x_residue.abs().mean())
        self.log("val mean absolute y_residue", y_residue.abs().mean())
        self.log("val mean absolute z_residue", z_residue.abs().mean())

        return loss

    @staticmethod
    def get_output_names():
        return [
            "x_residue",
            "y_residue",
            "z_residue",
            "infered cx",
            "infered cy",
            "infered cz",
            "infered thrust",
        ]
