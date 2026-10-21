############################## Importações ##############################
# Bibliotetecas gerais
import pandas as pd
import numpy as np

# Modelagem
    # Algoritmos
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svc import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    CategoricalNB,
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
from sklean.preprocessing import StandardScaler

import optuna
from optuna import create_study
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Semente aleatória
RANDOM_SEED = 3001

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
y_treino_b = df_treino_b.reindex(TARGET).values.ravel()

############################# Instancias #############################
def cria_instancia_kNN(trial):
    """Cria uma instância do modelo de k-NN.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.
    """

    parametros = {
        # Espaço de busca
        "n_neighbors": trial.suggest_int("n_neighbors", 5, 500, log=True),
        "wights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_float("p", 1.0, 2),
        # Argumentos fixos
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
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
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.05, log=True),
        "min_samples_leaf": trial.suggest_float("mins_samples_leaf", 0.001, 0.05, log=True),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0,5),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 6, 200, log=True),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.5),
        # Argumentos fixos
        "class_weight": "balanced",
        "n_jobs": -1,
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
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.05, log=True),
        "min_samples_leaf": trial.suggest_float("mins_samples_leaf", 0.001, 0.05, log=True),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0,5),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 6, 200, log=True),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.5),
        "class_weight": trial.suggest_categorical("class_weight", "balanced", "balanced_subsample"),
        # Argumentos fixos
        "n_jobs": -1,
        "bootstrap": True,
        "random_state": RANDOM_SEED,
    }

    n_features = trial.suggest_int("n_components", 10, 18)

    modelo = make_pipeline(
        PCA(n_features),
        DecisionTreeClassifier(**parametros)
    )

    return modelo

def cria_instancia_kNN_ensemble(trial):
    """Cria uma instância do modelo de floresta aleatória.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.
    """

    parametros_kNN = {
        # Espaço de busca
        "n_neighbors": trial.suggest_int("n_neighbors", 5, 500, log=True),
        "wights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_float("p", 1.0, 2),
        #Argumentos fixos
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    parametros_ensamble = {
        # Espaço de busca
        "n_estimators": trial.suggest_int("n_estimators", 10, 500, log=True),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        # Argumentos fixos
        "estimator": KNeighborsClassifier(**parametros_kNN),
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    n_features = trial.suggest_int("n_components", 10, 15)

    modelo = make_pipeline(
        PCA(n_features),
        StandardScaler(),
        BaggingClassifier(**parametros_kNN)
    )

    return modelo

def cria_instancia_HGradientBoosting(trial):
    """Cria uma instância do modelo de floresta aleatória.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.
    """

    parametros = {
        # Espaço de busca
        "learning_rate": trial.suggest_float("learning_rate", 0.0, 1.0),
        "max_iter": trial.suggest_int("max_iter", 100, 300),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 6, 150, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 100, 500),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 9.0),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        "max_bins": trial.suggest_int("max_bins", 100, 400),

        # Argumentos fixos
        "validation_fraction": 0.1,
        "verbose": 1,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    n_features = trial.suggest_int("n_components", 10, 18)

    modelo = make_pipeline(
        PCA(n_features),
        HistGradientBoostingClassifier(**parametros)
    )

    return modelo

def cria_instancia_NB(trial):
    """Cria uma instância do modelo de floresta aleatória.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.
    """

    parametros = {
        # Espaço de busca
        "alpha": trial.suggest_float("alpha", 0.0, 1.0),
        # Argumentos fixos
        "fit_prior": True,
        "force_alpha": True,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    n_features = trial.suggest_int("n_components", 10, 18)

    modelo = make_pipeline(
        PCA(n_features),
        CategoricalNB(**parametros)
    )

    return modelo

def cria_instancia_Stacking(trial):
    """Cria uma instância do modelo de floresta aleatória.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.
    """

    parametros_LinearSVC = {
      # Espaço de busca
      "penality": trial.suggest_categorical("penality", ["l1", "l2"]),
      "loss": trial.suggest_categorical("loss", ['hinge', 'squared_hinge']),
      "tol": trial.suggest_float("tol", 1e-4, 0.01, log=True),
      "C": trial.suggest_float("C", 1e-3, 5, log=True),
      "max_iter": trial.suggest_int("max_iter", 500, 5000, log=True),
      # Argumentos fixos
      "dual": "auto",
      "multi_class": "ovr",
      "class_weight": "balanced",
      "n_jobs": -1,
      "random_state": RANDOM_SEED,
    }

    parametros_forest = {
        # Espaço de busca
        "n_estimators": trial.suggest_int("n_estimators", 10, 300, log=True),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.05, log=True),
        "min_samples_leaf": trial.suggest_float("mins_samples_leaf", 0.001, 0.05, log=True),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0,5),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 6, 200, log=True),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.5),
        "class_weight": trial.suggest_categorical("class_weight", "balanced", "balanced_subsample"),
        # Argumentos fixos
        "n_jobs": -1,
        "bootstrap": True,
        "random_state": RANDOM_SEED,
    }

    estimators = [
      ("lsvc", LinearSVC(**parametros_LinearSVC)),
      ("rfc", RandomForestClassifier(**parametros_forest ))
    ]

    parametros_stacking = {
        # Espaço de busca
        "stack_method": trial.suggest_categorical("stack_method", ['auto', 'predict_proba', 'decision_function', 'predict']),
        # Argumentos fixos
        "cv": StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    n_features = trial.suggest_int("n_components", 10, 18)

    modelo = make_pipeline(
        PCA(n_features),
        StackingClassifier(**parametros_stacking)
    )

    return modelo


############################## Funções Objetivo ###########################

# Definindo métrica ROC-AUC
ras = make_scorer(roc_auc_score, average="weighted", response_method="predict_proba")

def funcao_objetivo_kNN(trial, X, y, n_folds):
    """Função objetivo do optuna

    Referencia:
      https://medium.com/@walter_sperat/ using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451
    """

    modelo = cria_instancia_kNN(trial)

    metricas = cross_val_score(
        modelo,
        X,
        y,
        scoring=ras,
        cv=n_folds
    )

    return metricas.mean()


def funcao_objetivo_tree(trial, X, y, n_folds):
    """Função objetivo do optuna

    Referencia:
      https://medium.com/@walter_sperat/ using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451
    """

    modelo = cria_instancia_tree(trial)

    metricas = cross_val_score(
        modelo,
        X,
        y,
        scoring=ras,
        cv=n_folds
    )

    return metricas.mean()

def funcao_objetivo_forest(trial, X, y, n_folds):
    """Função objetivo do optuna

    Referencia:
      https://medium.com/@walter_sperat/ using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451
    """

    modelo = cria_instancia_forest(trial)

    metricas = cross_val_score(
        modelo,
        X,
        y,
        scoring=ras,
        cv=n_folds
    )

    return metricas.mean()

def funcao_objetivo_kNN_ensemble(trial, X, y, n_folds):
    """Função objetivo do optuna

    Referencia:
      https://medium.com/@walter_sperat/ using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451
    """

    modelo = cria_instancia_kNN_ensemble(trial)

    metricas = cross_val_score(
        modelo,
        X,
        y,
        scoring=ras,
        cv=n_folds
    )

    return metricas.mean()

def funcao_objetivo_HGradientBoosting(trial, X, y, n_folds):
    """Função objetivo do optuna

    Referencia:
      https://medium.com/@walter_sperat/ using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451
    """

    modelo = cria_instancia_HGradientBoosting(trial)

    metricas = cross_val_score(
        modelo,
        X,
        y,
        scoring=ras,
        cv=n_folds
    )

    return metricas.mean()

def funcao_objetivo_NB(trial, X, y, n_folds):
    """Função objetivo do optuna

    Referencia:
      https://medium.com/@walter_sperat/ using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451
    """

    modelo = cria_instancia_NB(trial)

    metricas = cross_val_score(
        modelo,
        X,
        y,
        scoring=ras,
        cv=n_folds
    )

    return metricas.mean()

def funcao_objetivo_Stacking(trial, X, y, n_folds):
    """Função objetivo do optuna

    Referencia:
      https://medium.com/@walter_sperat/ using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451
    """

    modelo = cria_instancia_Stacking(trial)

    metricas = cross_val_score(
        modelo,
        X,
        y,
        scoring=ras,
        cv=n_folds
    )

    return metricas.mean()

############################## Creat studies ###########################

NOME_ESTUDO_kNN = "YSO_classifier-kNN"

objeto_de_estudo = create_study(
    direction="minimize",
    study_name=NOME_ESTUDO_kNN,
    storage=f"sqlite:///{NOME_ESTUDO_kNN}.db",
    load_if_exists=True,
)

NOME_ESTUDO_tree = "YSO_classifier-Tree"

objeto_de_estudo = create_study(
    direction="minimize",
    study_name=NOME_ESTUDO_tree,
    storage=f"sqlite:///{NOME_ESTUDO_tree}.db",
    load_if_exists=True,
)

NOME_ESTUDO_forest = "YSO_classifier-Forest"

objeto_de_estudo = create_study(
    direction="minimize",
    study_name=NOME_ESTUDO_forest,
    storage=f"sqlite:///{NOME_ESTUDO_forest}.db",
    load_if_exists=True,
)

NOME_ESTUDO_kNN_ensemble = "YSO_classifier-kNN_ensemble"

objeto_de_estudo = create_study(
    direction="minimize",
    study_name=NOME_ESTUDO_kNN_ensemble,
    storage=f"sqlite:///{NOME_ESTUDO_kNN_ensemble}.db",
    load_if_exists=True,
)

NOME_ESTUDO_Hist_Gradient_Boosting = "YSO_classifier-Hist_Gradient_Boosting"

objeto_de_estudo = create_study(
    direction="minimize",
    study_name=NOME_ESTUDO_Hist_Gradient_Boosting,
    storage=f"sqlite:///{NOME_ESTUDO_Hist_Gradient_Boosting}.db",
    load_if_exists=True,
)

NOME_ESTUDO_Naive_Bayes = "YSO_classifier-Naive_Bayes"

objeto_de_estudo = create_study(
    direction="minimize",
    study_name=NOME_ESTUDO_Naive_Bayes,
    storage=f"sqlite:///{NOME_ESTUDO_Naive_Bayes}.db",
    load_if_exists=True,
)

NOME_ESTUDO_Stacking = "YSO_classifier-Stacking-Linear_SVC_and_Forest"

objeto_de_estudo = create_study(
    direction="minimize",
    study_name=NOME_ESTUDO_Stacking,
    storage=f"sqlite:///{NOME_ESTUDO_Stacking}.db",
    load_if_exists=True,
)




















