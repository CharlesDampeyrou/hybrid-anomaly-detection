"""
Auteur : Charles Dampeyrou
licence : Creative Commons BY-NC 4.0

Ce script permet de lancer l'entraînement du modèle hybride de detection d'anomalies.

Avant de lancer ce script, il est nécessaire de :
- Adapter le chargement des données (lignes 118 à 121)
- Préciser le chemin vers le répertoire de sauvegarde du modèle (ligne 219)
- Adapter les noms des variables d'entrée du modèle (lignes 58 à 82)
- Installer toutes les dépendances nécessaires (cf README.md)

Lors de l'entrainement, il est possible de suivre la progression des métriques à l'aide de
TensorBoard. Pour cela, il est nécessaire d'installer TensorBoard et de lancer la commande
suivante dans un terminal :
tensorboard --logdir=logs

Si la taille des donneés est trop importante et que l'entrainement est trop lent, il est possible
de limiter le nombre de données utilisées pour l'entrainement en modifiant l'argument
data_reduction (ligne 164). Par exemple, data_reduction=0.1 permet d'utiliser 10% des données.

Il est possible de réduire la RAM nécessaire au détriment de la vitesse d'entrainement en utilisant
la classe GenericFlightMecaDataset à la place de GenericFlightMecaDatasetFast (ligne 242).

Il est possible de préciser la surface ailaire de l'avion ligne 215 pour obtenir les coefficients
de portance et de traînée. Attention cependant à utiliser la même valeur pour l'entrainement et
l'inférence.

Il est possible de préciser quels vols utiliser pour l'entrainement et pour le test en précisant
la variable train_flight_names (ligne 150, à préciser aussi pour l'inférence le cas échéant). 
"""

####################################################################################################
# IMPORTS
####################################################################################################

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from reader import FromPandasDataset
from model import (
    PhysicalModel,
    FlightMecaNet3DEq,
    GenericFlightMecaDataset,
    GenericFlightMecaDatasetFast,
)
from reader.preprocessing import lbs_to_kg, deg_to_rad, celcius_to_kelvin, kts_to_ms

####################################################################################################
# VARIABLES
################################################################################################

training_name = "test_0"

time_var = "Time, s"
flight_name_var = "flight_name"
mass_var = "MASS"
jx_var = "JX"
jy_var = "JY"
jz_var = "JZ"
alpha_var = "ALPHA"
beta_var = "BETA"
pressure_var = "PST"
temp_var = "SAT"
air_speed_var = "VTAS"
mach_var = "MACH"
gear_var = "MLG"
flaps_bool_var = "DB"
flaps_var = "DVOLIG"
stab_var = "TRIM"
elevator_var = "DM"
rudder_var = "DN"
aileron_var = "DL"
spoiler_var = "DSPOIL"
n1_var = "N1"
altitude_var = "ZBPIL"
roll_angle_var = "PHI"

cx_input_vars = [
    alpha_var,
    flaps_var,
    stab_var,
    elevator_var,
    mach_var,
    gear_var,
]

cy_input_vars = [
    beta_var,
    rudder_var,
    mach_var,
]

cz_input_vars = [
    alpha_var,
    flaps_var,
    stab_var,
    elevator_var,
    mach_var,
    gear_var,
]

thrust_input_vars = [
    n1_var,
    pressure_var,
    temp_var,
]

####################################################################################################
# CHARGEMENT DES DONNÉES
####################################################################################################

datapath = Path.cwd() / "test_data" / "test_data.csv"

df = pd.read_csv(datapath)
df = df.set_index([flight_name_var, time_var])

####################################################################################################
# CONVERTION DES UNITÉS
####################################################################################################

df = lbs_to_kg(df, [mass_var])
df = deg_to_rad(
    df,
    [
        alpha_var,
        beta_var,
        elevator_var,
        rudder_var,
        aileron_var,
        spoiler_var,
        roll_angle_var,
    ],
)
df = celcius_to_kelvin(df, [temp_var])
df = kts_to_ms(df, [air_speed_var])

####################################################################################################
# CRÉATION DU DATASET
####################################################################################################

to_normalize = cx_input_vars + cy_input_vars + cz_input_vars + thrust_input_vars
to_normalize = list(set(to_normalize))

# train_flight_names = [
#     "vol 1",
#     "vol 2",
#     "vol 3",
#     "vol 4",
#     "vol 5",
#     "vol 6",
#     "vol 7",
#     "vol 8",
# ]
train_flight_names = None  # Répartition par défaut des vols en entrainement et test
data = FromPandasDataset(
    df,
    train_flight_names=train_flight_names,
    data_reduction=None,
    to_normalize=to_normalize,
    filter_train_phases=True,
)

####################################################################################################
# DÉFINITION DES HYPERPARAMÈTRES ET INITIALISATION DU MODÈLE
####################################################################################################

dataset_var_names_list = [
    cx_input_vars,
    cy_input_vars,
    cz_input_vars,
    thrust_input_vars,
]
dataset_var_names = [
    mass_var,
    jx_var,
    jy_var,
    jz_var,
    alpha_var,
    beta_var,
    pressure_var,
    temp_var,
    air_speed_var,
]
dataset_params = {
    "var_names": dataset_var_names,
    "var_names_list": dataset_var_names_list,
}


net_coef_dict = {
    "cx_net_coef": 1e-2,
    "cy_net_coef": 1e-3,
    "cz_net_coef": 1e-1,
    "trust_net_coef": 1e3,
}

net_params = {
    "cx_param_dim": len(cx_input_vars),
    "cy_param_dim": len(cy_input_vars),
    "cz_param_dim": len(cz_input_vars),
    "thrust_param_dim": len(thrust_input_vars),
    "regressor_layers": 3,
    "regressor_layer_dim": 64,
    "lr": 1e-5,
    "net_coef_dict": net_coef_dict,
    "equation_params": {
        "air_molar_mass": 29e-3,
        "gas_constant": 8.314,
        "wing_surface": 21.5,  # surface ailaire, en m^2
    },
}

saving_dir = Path.cwd() / "checkpoints" / training_name
log_dir = Path.cwd() / "logs"

callbacks = [
    ModelCheckpoint(
        saving_dir,
        "checkpoint.ckpt",
        monitor="val loss",
    )
]

trainer_params = {
    "max_epochs": 200,
    "callbacks": callbacks,
    "logger": TensorBoardLogger(save_dir=log_dir),
}

model = PhysicalModel(
    log_dir=log_dir,
    saving_dir=saving_dir,
    saving_name="checkpoint.ckpt",
    NetClass=FlightMecaNet3DEq,
    net_params=net_params,
    DatasetClass=GenericFlightMecaDatasetFast,
    dataset_params=dataset_params,
    trainer_params=trainer_params,
    batch_size=1024,
    num_loader_workers=8,  # à ajuster selon le nombre de coeurs dispos
)

####################################################################################################
# ENTRAINEMENT DU MODÈLE
####################################################################################################

if not model._is_fitted:
    model.fit(data)
else:
    print("Modèle déjà entrainé !")
