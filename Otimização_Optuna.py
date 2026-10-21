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
from sklearn.linear_model import RidgeClassifierCV
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
from optuna.samplers import GPSampler
from functools import partial

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
        "p": trial.suggest_float("p", 1.0, 2, step=0.01),
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
        "max_depth": trial.suggest_int("max_depth", 3, 100, log=True),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.05, log=True),
        "min_samples_leaf": trial.suggest_float("mins_samples_leaf", 0.001, 0.05, log=True),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5, step=0.1),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 6, 200, log=True),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.5, step=0.1),
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
        "max_depth": trial.suggest_int("max_depth", 3, 100, log=True),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.05, log=True),
        "min_samples_leaf": trial.suggest_float("mins_samples_leaf", 0.001, 0.05, log=True),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0,5, step=0.1),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0, step=0.05),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 6, 200, log=True),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.5, step=0.1),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
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
        "p": trial.suggest_float("p", 1.0, 2, step=0.01),
        #Argumentos fixos
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    parametros_ensamble = {
        # Espaço de busca
        "n_estimators": trial.suggest_int("n_estimators", 10, 500, log=True),
        "max_samples": trial.suggest_float("max_samples", 0.7, 1.0, step=0.05),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0, step=0.1),
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
        "learning_rate": trial.suggest_float("learning_rate", 0.0, 1.0, step=0.01),
        "max_iter": trial.suggest_int("max_iter", 100, 300, step=5),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 5, 150, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 100, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 100, 1000, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 9.0, step=0.5),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        "max_bins": trial.suggest_int("max_bins", 100, 400, step=10),

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
        "alpha": trial.suggest_float("alpha", 0.0, 5.0, step=0.1),
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
      "C": trial.suggest_float("C", 1e-3, 5.0, log=True),
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
        "max_depth": trial.suggest_int("max_depth", 3, 100, log=True),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.05, log=True),
        "min_samples_leaf": trial.suggest_float("mins_samples_leaf", 0.001, 0.05, log=True),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5, step=0.1),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0, step=0.1),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 6, 200, log=True),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.5, step=0.1),
        "class_weight": trial.suggest_categorical("class_weight", "balanced", "balanced_subsample"),
        # Argumentos fixos
        "n_jobs": -1,
        "bootstrap": True,
        "random_state": RANDOM_SEED,
    }

    parametros_NB = {
        # Espaço de busca
        "alpha": trial.suggest_float("alpha", 0.0, 1.0, step=0.05),
        # Argumentos fixos
        "fit_prior": True,
        "force_alpha": True,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    estimators = [
      ("lsvc", LinearSVC(**parametros_LinearSVC)),
      ("rfc", RandomForestClassifier(**parametros_forest ))
      ("cnb", CategoricalNB(**parametros_NB))
    ]

    alphas_ridge = np.array([
        trial.suggest_float(f"alpha {i}", 1e-3, 1.0, log=True)
        for i in range(3)
    ])

    parametros_ridge_CV = {
        # Espaço de busca
        "alphas": alphas_ridge,
        # Argumentos fixos
        "fit_intercept": True,
        "scoring": ras,
        "cv": StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
        "class_weight": "balanced",
    }

    multicamada = trial.suggest("multilayer", True, False)

    parametros_stacking = {
        # Espaço de busca
        "stack_method": trial.suggest_categorical("stack_method", ['auto', 'predict_proba', 'decision_function', 'predict']),
        "passtrhough": trial.suggest_categorical("passtrhough", True, False),
        # Argumentos fixos
        "estimators": estimators,
        "final_estimator": RidgeClassifierCV(**parametros_ridge_CV),
        "cv": StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    if multicamada:
        parametros_ultimate_stacking = parametros_stacking
        parametros_stacking["final_estimator"] = StackingClassifier(**parametros_ultimate_stacking)

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

objeto_de_estudo_kNN = create_study(
    direction="minimize",
    study_name=NOME_ESTUDO_kNN,
    storage=f"sqlite:///{NOME_ESTUDO_kNN}.db",
    load_if_exists=True,
    sampler=GPSampler(),
)

NOME_ESTUDO_tree = "YSO_classifier-Tree"

objeto_de_estudo_tree = create_study(
    direction="minimize",
    study_name=NOME_ESTUDO_tree,
    storage=f"sqlite:///{NOME_ESTUDO_tree}.db",
    load_if_exists=True,
    sampler=GPSampler(),    
)

NOME_ESTUDO_forest = "YSO_classifier-Forest"

objeto_de_estudo_forest = create_study(
    direction="minimize",
    study_name=NOME_ESTUDO_forest,
    storage=f"sqlite:///{NOME_ESTUDO_forest}.db",
    load_if_exists=True,
    sampler=GPSampler(),
)

NOME_ESTUDO_kNN_ensemble = "YSO_classifier-kNN_ensemble"

objeto_de_estudo_kNN_ensemble = create_study(
    direction="minimize",
    study_name=NOME_ESTUDO_kNN_ensemble,
    storage=f"sqlite:///{NOME_ESTUDO_kNN_ensemble}.db",
    load_if_exists=True,
    sampler=GPSampler(),
)

NOME_ESTUDO_Hist_Gradient_Boosting = "YSO_classifier-Hist_Gradient_Boosting"

objeto_de_estudo_HGradientBoosting = create_study(
    direction="minimize",
    study_name=NOME_ESTUDO_Hist_Gradient_Boosting,
    storage=f"sqlite:///{NOME_ESTUDO_Hist_Gradient_Boosting}.db",
    load_if_exists=True,
    sampler=GPSampler(),
)

NOME_ESTUDO_Naive_Bayes = "YSO_classifier-Naive_Bayes"

objeto_de_estudo_NB = create_study(
    direction="minimize",
    study_name=NOME_ESTUDO_Naive_Bayes,
    storage=f"sqlite:///{NOME_ESTUDO_Naive_Bayes}.db",
    load_if_exists=True,
    sampler=GPSampler(),
)

NOME_ESTUDO_Stacking = "YSO_classifier-Stacking-Linear_SVC_and_Forest_and_Cat_NB-Ridge_CV-Multilayer_or_not"

objeto_de_estudo_Stacking = create_study(
    direction="minimize",
    study_name=NOME_ESTUDO_Stacking,
    storage=f"sqlite:///{NOME_ESTUDO_Stacking}.db",
    load_if_exists=True,
    sampler=GPSampler(),
)

############################## Partial functions ###########################

NUM_FOLDS=100

funcao_objetivo_parcial_kNN = partial(
    funcao_objetivo_kNN, X=X_treino, y=y_treino, num_folds=NUM_FOLDS
)

funcao_objetivo_parcial_tree = partial(
    funcao_objetivo_tree, X=X_treino, y=y_treino, num_folds=NUM_FOLDS
)

funcao_objetivo_parcial_forest = partial(
    funcao_objetivo_forest, X=X_treino, y=y_treino, num_folds=NUM_FOLDS
)

funcao_objetivo_parcial_kNN_ensemble = partial(
    funcao_objetivo_kNN_ensemble, X=X_treino, y=y_treino, num_folds=NUM_FOLDS
)

funcao_objetivo_parcial_HGradientBoosting = partial(
    funcao_objetivo_HGradientBoosting, X=X_treino, y=y_treino, num_folds=NUM_FOLDS
)

funcao_objetivo_parcial_NB = partial(
    funcao_objetivo_NB, X=X_treino, y=y_treino, num_folds=NUM_FOLDS
)

funcao_objetivo_parcial_Stacking = partial(
    funcao_objetivo_Stacking, X=X_treino, y=y_treino, num_folds=NUM_FOLDS
)

############################## Forçando trials ###########################

# kNN
objeto_de_estudo_kNN.enqueue_trial({
    "n_neighbors": 5,
    "weights": "uniform",
    "p": 2
})

# Tree
objeto_de_estudo_tree.enqueue_trial({
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0
})

# Forest
objeto_de_estudo_forest.enqueue_trial({
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced"
})

objeto_de_estudo_forest.enqueue_trial({
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced_subsample"
})

# kNN_ensemble
objeto_de_estudo_kNN_ensemble.enqueue_trial({
    "n_neighbors": 5,
    "weights": "uniform",
    "p": 2,

    "n_estimators": 10,
    "max_samples": 1.0,
    "max_features": 1.0
})

objeto_de_estudo_kNN_ensemble.enqueue_trial({
    "n_neighbors": 5,
    "weights": "uniform",
    "p": 2,

    "n_estimators": 10,
    "max_samples": 1.0,
    "max_features": 0.33
})

objeto_de_estudo_kNN_ensemble.enqueue_trial({
    "n_neighbors": 5,
    "weights": "uniform",
    "p": 2,

    "n_estimators": 100,
    "max_samples": 1.0,
    "max_features": 1.0
})

objeto_de_estudo_kNN_ensemble.enqueue_trial({
    "n_neighbors": 5,
    "weights": "uniform",
    "p": 2,

    "n_estimators": 100,
    "max_samples": 1.0,
    "max_features": 0.33
})

# HistGradientBoosting
objeto_de_estudo_HGradientBoosting.enqueue_trial({
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": 100,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_features": 1.0,
    "max_bins": 255
})

objeto_de_estudo_HGradientBoosting.enqueue_trial({
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": 100,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_features": 0.33,
    "max_bins": 255
})

# CategoricalNaiveBayes
objeto_de_estudo_NB.enqueue_trial({
    "alpha": 1.0
})

# Stackig
objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": False,
    "stack_method": "auto",
    "passtrhough": False
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": False,
    "stack_method": "predict_proba",
    "passtrhough": False
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": False,
    "stack_method": "auto",
    "passtrhough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced_subsample",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": False,
    "stack_method": "auto",
    "passtrhough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced_subsample",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": False,
    "stack_method": "predict_proba",
    "passtrhough": False
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced_subsample",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": False,
    "stack_method": "auto",
    "passtrhough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": False,
    "stack_method": "predict_proba",
    "passtrhough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced_subsample",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": False,
    "stack_method": "predict_proba",
    "passtrhough": True
})

    # Multicamada

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": True,
    "stack_method": "auto",
    "passtrhough": False
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": True,
    "stack_method": "predict_proba",
    "passtrhough": False
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": True,
    "stack_method": "auto",
    "passtrhough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced_subsample",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": True,
    "stack_method": "auto",
    "passtrhough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced_subsample",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": True,
    "stack_method": "predict_proba",
    "passtrhough": False
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced_subsample",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": True,
    "stack_method": "auto",
    "passtrhough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": True,
    "stack_method": "predict_proba",
    "passtrhough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penality": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,
    "max_iter": 1000,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.33,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced_subsample",

    # Categorical Naive Bayes
    "alpha": 1.0,

    # Ridge Classifier CV
    "alphas": (0.1, 1.0, 10.0),

    # Stacking
    "multilayer": True,
    "stack_method": "predict_proba",
    "passtrhough": True
})
############################## Otimizando ###########################

NUM_TRIALS = 200

objeto_de_estudo_kNN.optimize(
    funcao_objetivo_parcial_kNN, n_trials=NUM_TRIALS/2
)

objeto_de_estudo_tree.optimize(
    funcao_objetivo_parcial_tree, n_trials=NUM_TRIALS/2
)

objeto_de_estudo_forest.optimize(
    funcao_objetivo_parcial_forest, n_trials=NUM_TRIALS
)

objeto_de_estudo_kNN_ensemble.optimize(
    funcao_objetivo_parcial_kNN_ensemble, n_trials=NUM_TRIALS
)

objeto_de_estudo_HGradientBoosting.optimize(
    funcao_objetivo_parcial_HGradientBoosting, n_trials=NUM_TRIALS
)

objeto_de_estudo_NB.optimize(
    funcao_objetivo_parcial_NB, n_trials=NUM_TRIALS
)

objeto_de_estudo_Stacking.optimize(
    funcao_objetivo_parcial_Stacking, n_trials=NUM_TRIALS
)

############################## Melhores trials ###########################

melhor_trial_kNN = objeto_de_estudo_kNN.best_trial

print(f"Número do melhor trial: {melhor_trial_kNN.number}")
print(f"Parâmetros do melhor trial: {melhor_trial_kNN.params}")

melhor_trial_tree = objeto_de_estudo_tree.best_trial

print(f"Número do melhor trial: {melhor_trial_tree.number}")
print(f"Parâmetros do melhor trial: {melhor_trial_tree.params}")

melhor_trial_forest = objeto_de_estudo_forest.best_trial

print(f"Número do melhor trial: {melhor_trial_forest.number}")
print(f"Parâmetros do melhor trial: {melhor_trial_forest.params}")

melhor_trial_kNN_ensamble = objeto_de_estudo_kNN_ensemble.best_trial

print(f"Número do melhor trial: {melhor_trial_kNN_ensamble.number}")
print(f"Parâmetros do melhor trial: {melhor_trial_kNN_ensamble.params}")

melhor_trial_HGradientBoosting = objeto_de_estudo_HGradientBoosting.best_trial

print(f"Número do melhor trial: {melhor_trial_HGradientBoosting.number}")
print(f"Parâmetros do melhor trial: {melhor_trial_HGradientBoosting.params}")

melhor_trial_NB = objeto_de_estudo_NB.best_trial

print(f"Número do melhor trial: {melhor_trial_NB.number}")
print(f"Parâmetros do melhor trial: {melhor_trial_NB.params}")

melhor_trial_Stacking = objeto_de_estudo_Stacking.best_trial

print(f"Número do melhor trial: {melhor_trial_Stacking.number}")
print(f"Parâmetros do melhor trial: {melhor_trial_Stacking.params}")