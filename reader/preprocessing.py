import numpy as np


def preprocess_train_test_df(df, to_normalize=None, data_reduction=None):
    if data_reduction is not None:
        interval = max(int(1 / data_reduction), 1)
        df = df.iloc[::interval]
    df, abs_max = normalize_columns(df, to_normalize)
    df.index.names = ["Flight name", "Time, s"]
    df = df.sort_values(by=["Flight name", "Time, s"])
    return df, abs_max


def normalize_columns(df, to_normalize):
    abs_max = None
    if to_normalize is not None:
        normalization_suffix = "_normalized"
        new_column_names = [col + normalization_suffix for col in to_normalize]
        abs_max = df[to_normalize].abs().max()
        abs_max[abs_max == 0] = 1  # avoid division by 0
        df[new_column_names] = df[to_normalize] / abs_max
    return df, abs_max


def convert_units(df, convertion_dict):
    for unit, columns in convertion_dict.items():
        if unit == "lbs":
            df = lbs_to_kg(df, columns)
        elif unit == "deg":
            df = deg_to_rad(df, columns)
        elif unit == "celcius":
            df = celcius_to_kelvin(df, columns)
        elif unit == "kts":
            df = kts_to_ms(df, columns)
    return df


def lbs_to_kg(df, lbs_columns):
    for col in lbs_columns:
        df[col] = df[col] * 0.453592
    return df


def deg_to_rad(df, deg_columns):
    for col in deg_columns:
        df[col] = df[col] * np.pi / 180
    return df


def celcius_to_kelvin(df, celcius_columns):
    for col in celcius_columns:
        df[col] = df[col] + 273.15
    return df


def kts_to_ms(df, kts_columns):
    for col in kts_columns:
        df[col] = df[col] * 0.514444
    return df
