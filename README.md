# aircraft-hybrid-AD

## Installation des dépendances

Afin d'installer les librairies nécessaires au fonctionnement du code, executer la commande suivante :

```
conda env create -f environment.yml
```

## Entraînement du modèle

Afin de pouvoir entrainer le modèle, il est nécessaire d'adapter le chargement des données, le chemin vers la sauvegarde du modèle ainsi que le nom des variables. L'ensemble des instructions est donné dans le script `training_script.py`.

Pour entraîner le modèle, se placer dans l'environement virtuel précédement créé (`conda activate hybrid-ad-env`), puis lancer le script d'entraînement avec la commande `python training_script.py`

## Visualisation des résultats

Une fois le modèle entraîné, il est possible de visualiser les prédictions du modèle à l'aide du notebook `result_explo.ipynb`. L'inférence étant plus rapide que l'entraînement, il est possible d'executer ce notebook sur une machine moins puissante que celle ayant servi à l'entraînement.

## Modifications pour la visualisation des données avec plotly

- Chargement des données : l'index du dataframe chargé doit être un MultiIndex avec en première valeur le nom du vol et en second index un DatetimeIndex qui indique l'instant considéré. cela peut être fait avec
```
df[time_var] = datetime(2025, 1, 1) + pd.to_timedelta(df[time_var], unit="s")
df = df.set_index([flight_name_var, time_var])
```

- Récupération des phases de vol : modification de la définition de `temp_df["vertical speed"]` pour gérer l'index en tant que DatetimeIndex plutôt qu'un index en secondes.