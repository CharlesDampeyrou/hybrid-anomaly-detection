import numpy as np
import pandas as pd
from pandas import IndexSlice
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_all_anomaly_scores(
    df,
    anomaly_detection_quantile,
):
    threshold = df["anomaly score"].quantile(anomaly_detection_quantile)
    is_bigger_than_threshold = lambda x: "r" if x > threshold else "b"
    color = df["anomaly score"].apply(is_bigger_than_threshold)
    nb_flights = df.index.get_level_values(0).unique().shape[0]
    fig, axes = plt.subplots(nrows=nb_flights, figsize=(10, nb_flights * 5))
    for idx, flight_name in enumerate(df.index.get_level_values(0).unique()):
        time = df.loc[flight_name, :].index
        score = df.loc[flight_name, :]["anomaly score"]
        axes[idx].scatter(time, score, c=color.loc[flight_name, :], marker=".")
        axes[idx].set_title(f"Vol : {flight_name}")
        axes[idx].set_xlabel("Time, s")
        axes[idx].set_ylabel("Anomaly score")
    return fig


def show_anomaly_score_single_flight(
    df,
    flight_name,
    start_time,
    end_time,
    anomaly_detection_quantile,
):
    to_show = df.loc[IndexSlice[flight_name, start_time:end_time], :]
    threshold = df["anomaly score"].quantile(anomaly_detection_quantile)
    detected_turbulence = np.logical_and(
        to_show["anomaly score"] > threshold,
        to_show["anomaly classification"] < 0,
    )
    detected_icing = np.logical_and(
        to_show["anomaly score"] > threshold,
        to_show["anomaly classification"] > 0,
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Anomaly score", "Anomaly classification"],
        x_title="Time, s",
        shared_xaxes=True,
    )
    fig.add_trace(
        go.Scatter(
            x=to_show.index.get_level_values(1),
            y=to_show["anomaly score"],
            name="Anomaly score",
            line={"color": "black"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=to_show.index.get_level_values(1),
            y=detected_turbulence,
            name="Detected_turbulence",
            line={"color": "red"},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=to_show.index.get_level_values(1),
            y=detected_icing,
            name="Detected_icing",
            line={"color": "blue"},
        ),
        row=2,
        col=1,
    )
    return fig


def show_residue_evolution(
    df,
    flight_name,
    start_time,
    end_time,
):
    to_show = df.loc[IndexSlice[flight_name, start_time:end_time], :]
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=["X-axis", "Y-axis", "Z-axis"],
        x_title="Time, s",
        shared_xaxes=True,
    )
    fig.add_trace(
        go.Scatter(
            x=to_show.index.get_level_values(1),
            y=to_show["x_residue"],
            name="X residue",
            line={"color": "blue"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=to_show.index.get_level_values(1),
            y=to_show["y_residue"],
            name="Y residue",
            line={"color": "red"},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=to_show.index.get_level_values(1),
            y=to_show["z_residue"],
            name="Z residue",
            line={"color": "green"},
        ),
        row=3,
        col=1,
    )
    return fig
