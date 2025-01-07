import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def create_dashboard(
    df,
    air_speed_var,
    altitude_var,
):
    """
    Visualisation des données de vol avec Dash.
    Entrée :
        - df : DataFrame contenant les données de vol. Le numéro de vol doit être le premier niveau de l'index et le temps en secondes au deuxième niveau. Si ce n'est pas le cas, les changements sont à faire dans la fonction update_graph et dans le layout au niveau de la définition du Dropdown.
        - air_speed_var : nom de la variable de vitesse air pour le graphe secondaire
        - altitude_var : nom de la variable d'altitude pour le graphe secondaire
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container(
        [
            html.H1("Dashboard exploration de vol", className="text-center my-4"),
            # Menu déroulant et boutons d'ajout
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Variable à ajouter :"),
                            dcc.Dropdown(
                                id="variable-dropdown",
                                options=[
                                    {"label": col, "value": col}
                                    for col in df.columns
                                    if col != "time"
                                ],
                                placeholder="Sélectionnez une variable",
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Vol :"),
                            dcc.Dropdown(
                                id="flight-dropdown",
                                options=[
                                    {"label": flight, "value": flight}
                                    for flight in df.index.get_level_values(0).unique()
                                ],
                                placeholder="Sélectionnez un vol",
                            ),
                        ],
                        width=3,
                    ),
                ],
                className="mb-4",
            ),
            # Liste des variables sélectionnées
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Variables selectionnées :"),
                            html.Div(id="selected-variables-list", className="mt-2"),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Graphique
            dbc.Row(
                [dbc.Col([dcc.Graph(id="variable-graph", style={"height": "500px"})])]
            ),
        ]
    )

    # Callbacks pour gérer la sélection des variables
    @app.callback(
        Output("selected-variables-list", "children"),
        Input("variable-dropdown", "value"),
        State("selected-variables-list", "children"),
    )
    def update_selected_variables(selected_variable, current_list):
        if selected_variable is None:
            raise dash.exceptions.PreventUpdate

        # Éviter les doublons
        if (current_list is not None) and any(
            selected_variable in item["props"]["children"][0] for item in current_list
        ):
            return current_list

        # Ajouter une variable à la liste
        print(selected_variable)
        new_item = dbc.ListGroupItem(
            [
                f"{selected_variable}",
                dbc.Button(
                    "✖",
                    id={"type": "remove-btn", "index": f"{selected_variable}"},
                    color="danger",
                    className="ml-3",
                    size="sm",
                ),
            ]
        )
        if current_list is None:
            return [new_item]
        else:
            return current_list + [new_item]

    # Callback pour supprimer une variable
    @app.callback(
        Output("selected-variables-list", "children", allow_duplicate=True),
        Input({"type": "remove-btn", "index": dash.ALL}, "n_clicks"),
        State("selected-variables-list", "children"),
        prevent_initial_call="initial_duplicate",
    )
    def remove_variable(n_clicks, current_list):
        if not n_clicks or all(click is None for click in n_clicks):
            raise dash.exceptions.PreventUpdate
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_list

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        triggered_variable = eval(triggered_id)["index"]

        # Filtrer la liste pour exclure la variable cliquée
        return [
            item
            for item in current_list
            if triggered_variable not in item["props"]["children"][0]
        ]

    # Callback pour mettre à jour le graphique
    @app.callback(
        Output("variable-graph", "figure"),
        Input("selected-variables-list", "children"),
        Input("flight-dropdown", "value"),
    )
    def update_graph(selected_variables, selected_flight):
        if not selected_variables or not selected_flight:
            return go.Figure()
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            specs=[[{}], [{"secondary_y": True}]],
        )
        for item in selected_variables:
            variable_name = item["props"]["children"][0]
            fig.add_trace(
                go.Scatter(
                    x=df.loc[selected_flight].index,
                    y=df.loc[selected_flight][variable_name],
                    mode="lines",
                    name=variable_name,
                    yaxis="y1",
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=df.loc[selected_flight].index,
                y=df.loc[selected_flight][altitude_var],
                mode="lines",
                name="Altitude, m",
                yaxis="y2",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.loc[selected_flight].index,
                y=df.loc[selected_flight][air_speed_var],
                mode="lines",
                name="Vitesse air, m/s",
                yaxis="y3",
            ),
            row=2,
            col=1,
            secondary_y=True,
        )
        fig.update_layout(
            yaxis1_title="",
            yaxis2_title="Altitude, m",
            yaxis3_title="Vitesse air, m/s",
            xaxis_title="Datetime",
        )
        return fig

    # Lancement de l'application
    app.run_server(host="127.0.0.1", debug=True)
