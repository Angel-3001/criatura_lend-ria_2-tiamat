############################## Importações ##############################
# Bibliotetecas gerais
import pandas as pd
import numpy as np
import copy
import os

# Modelagem
    # Algoritmos
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    BaggingClassifier,
    StackingClassifier
)
from sklearn.decomposition import PCA
    # Métricas
from sklearn.metrics import (
    confusion_matrix,
    make_scorer,
    roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
    # Normalização
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

import optuna
from optuna import create_study
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import GPSampler
from functools import partial

# Constantes
RANDOM_SEED = 3001
NUM_TRIALS = 100
NUM_FOLDS = 10

############################## Definições ##############################
ATRIBUTOS = [
    "ra", "dec", "l", "b", "silicate",
    "pah", "env", "mag3_6", "e_mag3_6",
    "mag4_5", "e_mag4_5", "mag5_8", "e_mag5_8", "mag8_0",
    "e_mag8_0", "csf", "m3_6", "m4_5", "m5_8", "m8_0"
]
TARGET = ["class"]

CLASSES = ["ClassI", "FS", "ClassII", "ClassIII"]

df_treino = pd.read_csv("YSOs-Dados_treino-Ordinal_encoder.csv")
df_treino_b = pd.read_csv("YSOs-Dados_treino-Ordinal_encoder-Balanceado.csv")

pesos_classes = np.array([
    len(df_treino)/(4 * len(df_treino.loc[df_treino["class"] == classe]))
    for classe in CLASSES
])

X_treino = df_treino.reindex(ATRIBUTOS, axis=1).values
y_treino = df_treino.reindex(TARGET, axis=1).values.ravel()

X_treino_b = df_treino_b.reindex(ATRIBUTOS, axis=1).values
y_treino_b = df_treino_b.reindex(TARGET, axis=1).values.ravel()

# Definindo métrica ROC-AUC
ras = make_scorer(roc_auc_score, average="weighted", response_method="predict_proba", multi_class="ovr")

############################## Criando instâncias ##############################

def cria_instancia_kNN(trial):
    """Cria uma instância do modelo de k-NN.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.
    """

    parametros = {
        # Espaço de busca
        "n_neighbors": trial.suggest_int("n_neighbors", 500, 5000, log=True),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_float("p", 1.0, 2, step=0.1),
        # Argumentos fixos
        "n_jobs": -1,
    }

    n_features = trial.suggest_int("n_components", 10, 15)

    modelo = make_pipeline(
        PCA(n_features),
        StandardScaler(),
        KNeighborsClassifier(**parametros)
    )

    return modelo

def cria_instancia_tree(trial):
    """Cria uma instância do modelo de árvore de decisão.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.
    """

    parametros = {
        # Espaço de busca
        "criterion": trial.suggest_categorical("criterion", ["entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 3, 150, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 200, 500),
        # Argumentos fixos
        "min_samples_split": 0.001,
        "min_samples_leaf": 0.001,
        "min_weight_fraction_leaf": 0.0,
        "min_impurity_decrease": 0.0,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    }

    n_features = trial.suggest_int("n_components", 10, 18)

    modelo = make_pipeline(
        PCA(n_features),
        DecisionTreeClassifier(**parametros)
    )

    return modelo

def cria_instancia_forest(trial):
    """Cria uma instância do modelo de floresta aleatória.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.
    """

    parametros = {
        # Espaço de busca
        "n_estimators": trial.suggest_int("n_estimators", 10, 300, log=True),
        "criterion": trial.suggest_categorical("criterion", ["entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 100, 500),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5, step=0.1),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0, step=0.05),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 200, 500),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.5, step=0.1),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        # Argumentos fixos
        "min_samples_split": 0.001,
        "min_samples_leaf": 0.001,
        "n_jobs": -1,
        "bootstrap": True,
        "random_state": RANDOM_SEED,
    }

    n_features = trial.suggest_int("n_components", 16, 18)

    modelo = make_pipeline(
        PCA(n_features),
        RandomForestClassifier(**parametros)
    )

    return modelo