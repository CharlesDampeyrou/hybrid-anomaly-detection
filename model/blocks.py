import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class SimpleRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nb_hidden_layers: int = 3,
        layers_dim: int = 32,
        last_activation=None,
        output_multiplier=1.0,
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, layers_dim)]
        layers.append(nn.LeakyReLU())
        for i in range(nb_hidden_layers):
            layers.append(nn.Linear(layers_dim, layers_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(layers_dim, 1))
        layers.append(nn.Flatten(start_dim=0))
        if last_activation == "Softplus":
            layers.append(nn.Softplus())
        elif last_activation == "ReLU":
            layers.append(nn.ReLU())
        elif last_activation == "LeakyReLU":
            layers.append(nn.LeakyReLU())
        elif last_activation == "Sigmoid":
            layers.append(nn.Sigmoid())
        elif last_activation != "no_activation" and last_activation is not None:
            raise ValueError(f"Unknown activation function {last_activation}")
        self.net = nn.Sequential(*layers)
        self.output_multiplier = output_multiplier

    def forward(self, x: torch.Tensor):
        return self.output_multiplier * self.net(x)
