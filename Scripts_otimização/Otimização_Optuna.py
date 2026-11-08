############################## ImportaÃ§Ãµes ##############################
# Bibliotetecas gerais
import pandas as pd     # Usado para carregar os arquivos.
import numpy as np      # Usado para trabalhar com arrays.
import copy             # Usado para copiar algumas coisas no Stacking.

# Modelagem
    # Algoritmos
from sklearn.pipeline import make_pipeline              # Cria pipeline.
from sklearn.neighbors import KNeighborsClassifier      # Classificador k-NN.
from sklearn.tree import DecisionTreeClassifier         # Árvore de Decisão classificadora.
from sklearn.svm import LinearSVC                       # SVM Linear SVC.
from sklearn.linear_model import LogisticRegression      # SVM Ridge CV.
from sklearn.naive_bayes import GaussianNB              # Algoritmo Naive-Bayes Gaussiano.
        # Algoritmos de comitê
from sklearn.ensemble import (
    RandomForestClassifier,             # Floresta Aleatória classificadora.
    HistGradientBoostingClassifier,     # Classificador Gradient Boosting por Histograma.
    BaggingClassifier,                  # Cria classificador por Bagging.
    StackingClassifier                  # Cria classificador por Stacking.
)
    # Outras coisas mais
from sklearn.model_selection import (
    cross_val_score,                                        # Validação cruzada.
    StratifiedKFold                                         # Cria k-Folds estratificados.
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler            # Normalizador padrão.
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA                       # Redução de dimensionalidade
                                                            #   por PCA.
from sklearn.calibration import CalibratedClassifierCV      # Calibração do Ridge CV, para
                                                            #   correção de algumas coisas.
    # Métricas
from sklearn.metrics import (
    roc_auc_score,              # Métrica ROC-AUC
    make_scorer                 # Usado para fazer uma métrica personalizada, corrigindo
                                #   certas peculiaridades.
)

# Optuna
import optuna                                           # Optuna.
from optuna import create_study                         # Cria o estudo.
optuna.logging.set_verbosity(optuna.logging.WARNING)    # Não mostra cada trial no log.
from optuna.samplers import GPSampler                   # Buscador Gaussiano.
from functools import partial                           # Usado para fazer a função objetivo
                                                        #   parcial.

# Semente aleatÃ³ria
RANDOM_SEED = 3001

############################## Definições ##############################
ATRIBUTOS = [
    "ra", "dec", "l", "b", "env", "mag3_6", "e_mag3_6",
    "mag4_5", "e_mag4_5", "mag5_8", "e_mag5_8", "mag8_0",
    "e_mag8_0", "c_mag36_45", "c_mag45_58", "c_mag58_80",
    "c_mag36_80", "csf", "m3_6", "m4_5", "m5_8", "m8_0"
]

ATRIBUTOS_NB = [
    "mag3_6", "e_mag3_6", "mag4_5", "e_mag4_5", "mag5_8", "e_mag5_8",
    "mag8_0", "e_mag8_0", "c_mag36_45", "c_mag45_58", "c_mag58_80", "c_mag36_80"
]

TARGET = ["class"]

CLASSES = ["ClassI", "FS", "ClassII", "ClassIII"]

df_treino = pd.read_csv("YSOs-Dados_treino-Ordinal_encoder.csv")
df_treino_b = pd.read_csv("YSOs-Dados_treino-Ordinal_encoder-Balanceado.csv")

pesos_classes = np.array([
    len(df_treino)/(4 * len(df_treino.loc[df_treino["class"] == classe]))
    for classe in CLASSES
])

X_treino = df_treino.reindex(ATRIBUTOS, axis=1)
y_treino = df_treino.reindex(TARGET, axis=1).values.ravel()

X_treino_b = df_treino_b.reindex(ATRIBUTOS, axis=1)
y_treino_b = df_treino_b.reindex(TARGET, axis=1).values.ravel()

# Definindo mÃ©trica ROC-AUC
ras = make_scorer(roc_auc_score, average="weighted", response_method="predict_proba", multi_class="ovr")

############################# Instancias #############################
def cria_instancia_kNN(trial):
    """Cria uma instÃ¢ncia do modelo de k-NN.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instÃ¢ncia do modelo desejado.
    """

    parametros = {
        # EspaÃ§o de busca
        "n_neighbors": trial.suggest_int("n_neighbors", 500, 5000, log=True),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_float("p", 1.0, 2, step=0.01),
    }

    n_features = trial.suggest_int("n_components", 10, 20)

    modelo = make_pipeline(
        PCA(n_features),
        StandardScaler(),
        KNeighborsClassifier(**parametros)
    )

    return modelo

def cria_instancia_tree(trial):
    """Cria uma instÃ¢ncia do modelo de Ã¡rvore de decisÃ£o.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instÃ¢ncia do modelo desejado.
    """

    parametros = {
        # EspaÃ§o de busca
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 3, 300, log=True),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.05, log=True),
        "min_samples_leaf": trial.suggest_float("mins_samples_leaf", 0.001, 0.05, log=True),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5, step=0.1),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 6, 500, log=True),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.5, step=0.1),
        # Argumentos fixos
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    }

    n_features = trial.suggest_int("n_components", 12, 20)

    modelo = make_pipeline(
        PCA(n_features),
        DecisionTreeClassifier(**parametros)
    )

    return modelo

def cria_instancia_forest(trial):
    """Cria uma instÃ¢ncia do modelo de floresta aleatÃ³ria.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instÃ¢ncia do modelo desejado.
    """

    parametros = {
        # EspaÃ§o de busca
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, log=True),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 3, 500, log=True),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.05, log=True),
        "min_samples_leaf": trial.suggest_float("mins_samples_leaf", 0.001, 0.05, log=True),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5, step=0.1),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0, step=0.05),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 6, 500, log=True),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.5, step=0.1),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        # Argumentos fixos
        "n_jobs": -1,
        "bootstrap": True,
        "random_state": RANDOM_SEED,
    }

    n_features = trial.suggest_int("n_components", 12, 20)

    modelo = make_pipeline(
        PCA(n_features),
        RandomForestClassifier(**parametros)
    )

    return modelo

def cria_instancia_kNN_ensemble(trial):
    """Cria uma instÃ¢ncia do modelo de floresta aleatÃ³ria.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instÃ¢ncia do modelo desejado.
    """

    parametros_kNN = {
        # EspaÃ§o de busca
        "n_neighbors": trial.suggest_int("n_neighbors", 500, 5000, log=True),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_float("p", 1.0, 2, step=0.01),
    }

    parametros_ensemble = {
        # EspaÃ§o de busca
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
        BaggingClassifier(**parametros_ensemble)
    )

    return modelo

def cria_instancia_HGradientBoosting(trial):
    """Cria uma instÃ¢ncia do modelo de floresta aleatÃ³ria.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instÃ¢ncia do modelo desejado.
    """

    parametros = {
        # EspaÃ§o de busca
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, step=0.01),
        "max_iter": trial.suggest_int("max_iter", 100, 500, step=5),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 5, 150, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 300, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 100, 1000, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 9.0, step=0.5),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        "max_bins": trial.suggest_int("max_bins", 100, 255, step=5),

        # Argumentos fixos
        "validation_fraction": 0.1,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    }

    n_features = trial.suggest_int("n_components", 10, 18)

    modelo = make_pipeline(
        PCA(n_features),
        HistGradientBoostingClassifier(**parametros)
    )

    return modelo

def cria_instancia_NB(trial):
    """Cria uma instÃ¢ncia do modelo de floresta aleatÃ³ria.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instÃ¢ncia do modelo desejado.
    """

    parametros = {
        # EspaÃ§o de busca
        "var_smoothing": trial.suggest_float("var_smoothing", 1e-9, 1e-1, log=True)
    }

    modelo = make_pipeline(
        ColumnTransformer([("fotometria", "passthrough", ATRIBUTOS_NB)], remainder="drop"),
        QuantileTransformer(output_distribution="normal"),
        GaussianNB(**parametros),
    )

    return modelo

def cria_instancia_Stacking(trial):
    """Cria uma instÃ¢ncia do modelo de floresta aleatÃ³ria.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instÃ¢ncia do modelo desejado.
    """
    
    n_features = trial.suggest_int("n_components", 10, 18)

    parametros_LinearSVC = {
      # EspaÃ§o de busca
      "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
      "tol": trial.suggest_float("tol", 1e-4, 0.01, log=True),
      "C": trial.suggest_float("C", 1e-3, 5.0, log=True),
      # Argumentos fixos
      "loss": "squared_hinge",
      "max_iter": 5000,
      "dual": False,
      "multi_class": "ovr",
      "class_weight": "balanced",
      "random_state": RANDOM_SEED,
    }

    parametros_forest = {
        # EspaÃ§o de busca
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, log=True),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 3, 500, log=True),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.05, log=True),
        "min_samples_leaf": trial.suggest_float("mins_samples_leaf", 0.001, 0.05, log=True),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5, step=0.1),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0, step=0.05),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 6, 500, log=True),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.5, step=0.1),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        # Argumentos fixos
        "n_jobs": -1,
        "bootstrap": True,
        "random_state": RANDOM_SEED,
    }

    parametros_NB = {
        # EspaÃ§o de busca
        "var_smoothing": trial.suggest_float("var_smoothing", 1e-9, 1e-1, log=True)
    }

    lsvc = CalibratedClassifierCV(
        make_pipeline(
            StandardScaler(),
            PCA(n_features),
            LinearSVC(**parametros_LinearSVC)
        ),
        cv=5
    )

    rfc = make_pipeline(
        PCA(n_features),
        RandomForestClassifier(**parametros_forest)
    )

    gnb = make_pipeline(
        ColumnTransformer([("fotometria", "passthrough", ATRIBUTOS_NB)], remainder="drop"),
        QuantileTransformer(output_distribution="normal"),
        GaussianNB(**parametros_NB),
    )

    estimators = [
      ("lsvc", lsvc),
      ("rfc", rfc),
      ("gnb", gnb)
    ]

    parametros_logistic_regression = {
        # EspaÃ§o de busca
        "C": trial.suggest_float("C_final", 1e-2, 10.0, log=True),
        # Argumentos fixos
        "class_weight": "balanced",
        "max_iter": 5000
    }

    multicamada = trial.suggest_categorical("multilayer", [True, False])

    parametros_stacking = {
        # EspaÃ§o de busca,
        "passthrough": trial.suggest_categorical("passthrough", [True, False]),
        # Argumentos fixos
        "stack_method": "predict_proba",
        "estimators": estimators,
        "final_estimator": LogisticRegression(**parametros_logistic_regression),
        "cv": StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
        "n_jobs": -1,
    }

    if multicamada:
        lr2 = make_pipeline(
            StandardScaler(), 
            LogisticRegression(**parametros_logistic_regression)
        )

        rfc2 = RandomForestClassifier(**parametros_forest)
        camada2 = [
            ("lr2", lr2),
            ("rf2", RandomForestClassifier(n_estimators=100, max_depth=6,
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)),
        ]
        parametros_stacking["final_estimator"] = StackingClassifier(
            estimators=camada2,
            final_estimator=LogisticRegression(**parametros_logistic_regression),
            stack_method="predict_proba",
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
            n_jobs=-1,
        )


    modelo = make_pipeline(
        StackingClassifier(**parametros_stacking)
    )

    return modelo


############################## FunÃ§Ãµes Objetivo ###########################

def funcao_objetivo_kNN(trial, X, y, n_folds):
    """FunÃ§Ã£o objetivo do optuna

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
    """FunÃ§Ã£o objetivo do optuna

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
    """FunÃ§Ã£o objetivo do optuna

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
    """FunÃ§Ã£o objetivo do optuna

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
    """FunÃ§Ã£o objetivo do optuna

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
    """FunÃ§Ã£o objetivo do optuna

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
    """FunÃ§Ã£o objetivo do optuna

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
    direction="maximize",
    study_name=NOME_ESTUDO_kNN,
    storage=f"sqlite:///{NOME_ESTUDO_kNN}.db",
    load_if_exists=True,
    sampler=GPSampler(),
)

NOME_ESTUDO_tree = "YSO_classifier-Tree"

objeto_de_estudo_tree = create_study(
    direction="maximize",
    study_name=NOME_ESTUDO_tree,
    storage=f"sqlite:///{NOME_ESTUDO_tree}.db",
    load_if_exists=True,
    sampler=GPSampler(),    
)

NOME_ESTUDO_forest = "YSO_classifier-Forest"

objeto_de_estudo_forest = create_study(
    direction="maximize",
    study_name=NOME_ESTUDO_forest,
    storage=f"sqlite:///{NOME_ESTUDO_forest}.db",
    load_if_exists=True,
    sampler=GPSampler(),
)

NOME_ESTUDO_kNN_ensemble = "YSO_classifier-kNN_ensemble"

objeto_de_estudo_kNN_ensemble = create_study(
    direction="maximize",
    study_name=NOME_ESTUDO_kNN_ensemble,
    storage=f"sqlite:///{NOME_ESTUDO_kNN_ensemble}.db",
    load_if_exists=True,
    sampler=GPSampler(),
)

NOME_ESTUDO_Hist_Gradient_Boosting = "YSO_classifier-Hist_Gradient_Boosting"

objeto_de_estudo_HGradientBoosting = create_study(
    direction="maximize",
    study_name=NOME_ESTUDO_Hist_Gradient_Boosting,
    storage=f"sqlite:///{NOME_ESTUDO_Hist_Gradient_Boosting}.db",
    load_if_exists=True,
    sampler=GPSampler(),
)

NOME_ESTUDO_Naive_Bayes = "YSO_classifier-Naive_Bayes"

objeto_de_estudo_NB = create_study(
    direction="maximize",
    study_name=NOME_ESTUDO_Naive_Bayes,
    storage=f"sqlite:///{NOME_ESTUDO_Naive_Bayes}.db",
    load_if_exists=True,
    sampler=GPSampler(),
)

NOME_ESTUDO_Stacking = "YSO_classifier-Stacking-Linear_SVC_and_Forest_and_Cat_NB-Ridge_CV-Multilayer_or_not"

objeto_de_estudo_Stacking = create_study(
    direction="maximize",
    study_name=NOME_ESTUDO_Stacking,
    storage=f"sqlite:///{NOME_ESTUDO_Stacking}.db",
    load_if_exists=True,
    sampler=GPSampler(),
)

############################## Partial functions ###########################

NUM_FOLDS=10

funcao_objetivo_parcial_kNN = partial(
    funcao_objetivo_kNN, X=X_treino, y=y_treino, n_folds=NUM_FOLDS
)

funcao_objetivo_parcial_tree = partial(
    funcao_objetivo_tree, X=X_treino, y=y_treino, n_folds=NUM_FOLDS
)

funcao_objetivo_parcial_forest = partial(
    funcao_objetivo_forest, X=X_treino, y=y_treino, n_folds=NUM_FOLDS
)

funcao_objetivo_parcial_kNN_ensemble = partial(
    funcao_objetivo_kNN_ensemble, X=X_treino, y=y_treino, n_folds=NUM_FOLDS
)

funcao_objetivo_parcial_HGradientBoosting = partial(
    funcao_objetivo_HGradientBoosting, X=X_treino, y=y_treino, n_folds=NUM_FOLDS
)

funcao_objetivo_parcial_NB = partial(
    funcao_objetivo_NB, X=X_treino, y=y_treino, n_folds=NUM_FOLDS
)

funcao_objetivo_parcial_Stacking = partial(
    funcao_objetivo_Stacking, X=X_treino, y=y_treino, n_folds=NUM_FOLDS//2
)

############################## ForÃ§ando trials ###########################

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
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0
})

# Forest
objeto_de_estudo_forest.enqueue_trial({
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
    "max_leaf_nodes": 200,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced"
})

objeto_de_estudo_forest.enqueue_trial({
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "max_features": 0.3
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
    "max_features": 0.3
})

# HistGradientBoosting
objeto_de_estudo_HGradientBoosting.enqueue_trial({
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": 100,
    "min_samples_leaf": 0.001,
    "l2_regularization": 0.0,
    "max_features": 1.0,
    "max_bins": 255
})

objeto_de_estudo_HGradientBoosting.enqueue_trial({
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": 100,
    "min_samples_leaf": 0.001,
    "l2_regularization": 0.0,
    "max_features": 0.3,
    "max_bins": 255
})

# CategoricalNaiveBayes
objeto_de_estudo_NB.enqueue_trial({
    "var_smoothing": 1e-9
})

# Stackig
objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": False
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": False
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": False
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": True
})

    # Multicamada

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": False
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": False
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": False
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": True
})

objeto_de_estudo_Stacking.enqueue_trial({
    # Linear CVS
    "penalty": "l2",
    "loss": "squared_hinge",
    "tol": 1e-4,
    "C": 1.0,

    # Random Forest
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 100,
    "min_samples_split": 0.001,
    "min_samples_leaf": 0.001,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.3,
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
    "passthrough": True
})
############################## Otimizando ###########################

NUM_TRIALS = 200

objeto_de_estudo_kNN.optimize(
    funcao_objetivo_parcial_kNN, n_trials=NUM_TRIALS//2
)

objeto_de_estudo_tree.optimize(
    funcao_objetivo_parcial_tree, n_trials=NUM_TRIALS//2
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

print(f"NÃºmero do melhor trial: {melhor_trial_kNN.number}")
print(f"ParÃ¢metros do melhor trial: {melhor_trial_kNN.params}")

melhor_trial_tree = objeto_de_estudo_tree.best_trial

print(f"NÃºmero do melhor trial: {melhor_trial_tree.number}")
print(f"ParÃ¢metros do melhor trial: {melhor_trial_tree.params}")

melhor_trial_forest = objeto_de_estudo_forest.best_trial

print(f"NÃºmero do melhor trial: {melhor_trial_forest.number}")
print(f"ParÃ¢metros do melhor trial: {melhor_trial_forest.params}")

melhor_trial_kNN_ensamble = objeto_de_estudo_kNN_ensemble.best_trial

print(f"NÃºmero do melhor trial: {melhor_trial_kNN_ensamble.number}")
print(f"ParÃ¢metros do melhor trial: {melhor_trial_kNN_ensamble.params}")

melhor_trial_HGradientBoosting = objeto_de_estudo_HGradientBoosting.best_trial

print(f"NÃºmero do melhor trial: {melhor_trial_HGradientBoosting.number}")
print(f"ParÃ¢metros do melhor trial: {melhor_trial_HGradientBoosting.params}")

melhor_trial_NB = objeto_de_estudo_NB.best_trial

print(f"NÃºmero do melhor trial: {melhor_trial_NB.number}")
print(f"ParÃ¢metros do melhor trial: {melhor_trial_NB.params}")

melhor_trial_Stacking = objeto_de_estudo_Stacking.best_trial

print(f"NÃºmero do melhor trial: {melhor_trial_Stacking.number}")
print(f"ParÃ¢metros do melhor trial: {melhor_trial_Stacking.params}")