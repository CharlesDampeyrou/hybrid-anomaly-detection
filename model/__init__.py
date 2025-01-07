from .physical_model import PhysicalModel
from .flight_meca_net_3d_eq import FlightMecaNet3DEq
from .datasets import (
    GenericFlightMecaDataset,
    GenericFlightMecaDatasetFast,
)

__all__ = [
    "PhysicalModel",
    "FlightMecaNet3DEq",
    "FlightMecaNet3DEqV2",
    "GenericFlightMecaDataset",
    "GenericFlightMecaDatasetFast",
]
