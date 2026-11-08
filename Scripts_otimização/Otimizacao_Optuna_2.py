############################## Importações ##############################
# Bibliotecas gerais
import os                   # Usado para checar se os bancos dos estudos antigos existem.
import pandas as pd         # Usado para carregar os arquivos.
import numpy as np          # Usado para trabalhar com arrays.

# Modelagem
    # Algoritmos
from sklearn.pipeline import make_pipeline              # Cria pipeline.
from sklearn.neighbors import KNeighborsClassifier      # Classificador k-NN.
from sklearn.tree import DecisionTreeClassifier         # Árvore de Decisão classificadora.
from sklearn.svm import LinearSVC                       # SVM Linear SVC.
from sklearn.linear_model import LogisticRegression     # Regressão logística (estimador final do Stacking).
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
    StratifiedKFold,                                        # Cria k-Folds estratificados.
    StratifiedGroupKFold                                    # k-Folds estratificados e por grupos.
)
from sklearn.compose import ColumnTransformer               # Seleciona colunas por modelo.
from sklearn.preprocessing import StandardScaler            # Normalizador padrão.
from sklearn.preprocessing import QuantileTransformer       # Deixa as colunas próximas de uma normal.
from sklearn.decomposition import PCA                       # Redução de dimensionalidade
                                                            #   por PCA.
from sklearn.calibration import CalibratedClassifierCV      # Calibração do LinearSVC, que não
                                                            #   tem predict_proba.
    # Métricas
from sklearn.metrics import (
    roc_auc_score,              # Métrica ROC-AUC
    make_scorer                 # Usado para fazer uma métrica personalizada, corrigindo
                                #   certas peculiaridades.
)

# Optuna
import optuna                                           # Optuna.
from optuna import create_study                         # Cria o estudo.
from optuna.trial import TrialState                     # Estados de um trial (COMPLETE, FAIL...).
from optuna.samplers import GPSampler                   # Buscador Gaussiano.
from functools import partial                           # Usado para fazer a função objetivo
                                                        #   parcial.

# Semente aleatória
RANDOM_SEED = 3001

############################## Definições ##############################
ATRIBUTOS = [
    "ra", "dec", "l", "b", "env", "mag3_6", "e_mag3_6",
    "mag4_5", "e_mag4_5", "mag5_8", "e_mag5_8", "mag8_0",
    "e_mag8_0", "c_mag36_45", "c_mag45_58", "c_mag58_80",
    "c_mag36_80", "csf", "m3_6", "m4_5", "m5_8", "m8_0"
]

# Atributos usados somente pelo Naive Bayes (fotometria contínua + cores)
ATRIBUTOS_NB = [
    "mag3_6", "e_mag3_6", "mag4_5", "e_mag4_5", "mag5_8", "e_mag5_8",
    "mag8_0", "e_mag8_0", "c_mag36_45", "c_mag45_58", "c_mag58_80", "c_mag36_80"
]

TARGET = "class"

CLASSES = ["ClassI", "FS", "ClassII", "ClassIII"]

# Sufixo dos estudos NOVOS. Os estudos antigos (sem sufixo) não são alterados:
# eles só são lidos para aproveitar os melhores hiperparâmetros como ponto de partida.
SUFIXO = "_v2"

df_treino = pd.read_csv("YSOs-Dados_treino-Ordinal_encoder.csv")

# Obs.: df[lista] dá KeyError se faltar alguma coluna (reindex criaria uma coluna toda NaN).
X_treino = df_treino[ATRIBUTOS]
y_treino = df_treino[TARGET].to_numpy(dtype="object")     # array 1-D (um DataFrame (n,1) zera todas as métricas)

grupos = df_treino.groupby(["group"]).ngroup().to_numpy()

print(type(X_treino))
print(type(y_treino))
print(type(grupos))

print(y_treino.dtype)

pesos_classes = np.array([
    len(df_treino)/(4 * len(df_treino.loc[df_treino[TARGET] == classe]))
    for classe in CLASSES
])

# Definindo métrica ROC-AUC
ras = make_scorer(roc_auc_score, average="weighted", response_method="predict_proba", multi_class="ovr")

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
    """Cria uma instância do modelo de árvore de decisão.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.
    """

    parametros = {
        # Espaço de busca
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 3, 300, log=True),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.05, log=True),
        "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.001, 0.05, log=True),
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
    """Cria uma instância do modelo de floresta aleatória.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.
    """

    parametros = {
        # Espaço de busca
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 3, 500, log=True),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.05, log=True),
        "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.001, 0.05, log=True),
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
    """Cria uma instância do modelo de k-NN com bagging.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.
    """

    parametros_kNN = {
        # Espaço de busca
        "n_neighbors": trial.suggest_int("n_neighbors", 500, 5000, log=True),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_float("p", 1.0, 2, step=0.01),
    }

    parametros_ensemble = {
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

    n_features = trial.suggest_int("n_components", 10, 20)

    modelo = make_pipeline(
        PCA(n_features),
        StandardScaler(),
        BaggingClassifier(**parametros_ensemble)
    )

    return modelo

def cria_instancia_HGradientBoosting(trial):
    """Cria uma instância do modelo de Gradient Boosting por histograma.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.
    """

    parametros = {
        # Espaço de busca
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, step=0.01),
        "max_iter": trial.suggest_int("max_iter", 100, 500, step=5),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 5, 200, log=True),
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
    """Cria uma instância do modelo de Naive Bayes Gaussiano (só fotometria + cores).

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.
    """

    parametros = {
        # Espaço de busca
        "var_smoothing": trial.suggest_float("var_smoothing", 1e-9, 1e-1, log=True)
    }

    modelo = make_pipeline(
        ColumnTransformer([("fotometria", "passthrough", ATRIBUTOS_NB)], remainder="drop"),
        QuantileTransformer(output_distribution="normal"),
        GaussianNB(**parametros),
    )

    return modelo

def cria_instancia_Stacking(trial):
    """Cria uma instância do modelo de Stacking (LinearSVC + Floresta + Naive Bayes -> Regressão Logística).

    O Naive Bayes enxerga só a fotometria; os outros estimadores veem todos os atributos.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.
    """

    n_features = trial.suggest_int("n_components", 10, 18)

    parametros_LinearSVC = {
      # Espaço de busca
      "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
      "tol": trial.suggest_float("tol", 1e-4, 0.01, log=True),
      "C": trial.suggest_float("C", 1e-3, 5.0, log=True),
      # Argumentos fixos (l1 só funciona com squared_hinge e dual=False)
      "loss": "squared_hinge",
      "max_iter": 5000,
      "dual": False,
      "class_weight": "balanced",
      "random_state": RANDOM_SEED,
    }

    parametros_forest = {
        # Espaço de busca
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, log=True),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 3, 500, log=True),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.05, log=True),
        "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.001, 0.05, log=True),
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
        # Espaço de busca
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
        # Espaço de busca (nome diferente do "C" do LinearSVC)
        "C": trial.suggest_float("C_final", 1e-2, 10.0, log=True),
        # Argumentos fixos
        "class_weight": "balanced",
        "max_iter": 5000
    }

    multicamada = trial.suggest_categorical("multilayer", [True, False])

    parametros_stacking = {
        # Espaço de busca
        "passthrough": trial.suggest_categorical("passthrough", [True, False]),
        # Argumentos fixos
        "stack_method": "predict_proba",
        "estimators": estimators,
        # O StandardScaler evita problemas de convergência quando passthrough=True
        "final_estimator": make_pipeline(
            StandardScaler(),
            LogisticRegression(**parametros_logistic_regression)
        ),
        "cv": StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
        "n_jobs": -1,
    }

    if multicamada:
        # A 2ª camada recebe só as probabilidades da 1ª (e os atributos brutos, sem nomes, se
        # passthrough=True), então precisa de estimadores próprios: copiar os da 1ª camada falha.
        camada2 = [
            ("lr2", make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=trial.suggest_float("C_camada2", 1e-2, 10.0, log=True),
                    class_weight="balanced", max_iter=5000
                )
            )),
            ("rf2", RandomForestClassifier(**parametros_forest)),
        ]
        parametros_stacking["final_estimator"] = StackingClassifier(
            estimators=camada2,
            final_estimator=parametros_stacking["final_estimator"],
            stack_method="predict_proba",
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
            n_jobs=-1,
        )

    modelo = StackingClassifier(**parametros_stacking)

    return modelo


############################## Funções Objetivo ###########################

def valida(modelo, X, y, grupos, n_folds):
    """Validação cruzada estratificada e por grupos (blocos de posição).

    Referencia:
      https://medium.com/@walter_sperat/ using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451
    """

    cv = StratifiedGroupKFold(n_folds, shuffle=True, random_state=RANDOM_SEED)

    metricas = cross_val_score(
        modelo,
        X,
        y,
        groups=grupos,
        scoring=ras,
        cv=cv
    )

    return metricas.mean()


def funcao_objetivo_kNN(trial, X, y, grupos, n_folds):
    """Função objetivo do optuna"""
    return valida(cria_instancia_kNN(trial), X, y, grupos, n_folds)

def funcao_objetivo_tree(trial, X, y, grupos, n_folds):
    """Função objetivo do optuna"""
    return valida(cria_instancia_tree(trial), X, y, grupos, n_folds)

def funcao_objetivo_forest(trial, X, y, grupos, n_folds):
    """Função objetivo do optuna"""
    return valida(cria_instancia_forest(trial), X, y, grupos, n_folds)

def funcao_objetivo_kNN_ensemble(trial, X, y, grupos, n_folds):
    """Função objetivo do optuna"""
    return valida(cria_instancia_kNN_ensemble(trial), X, y, grupos, n_folds)

def funcao_objetivo_HGradientBoosting(trial, X, y, grupos, n_folds):
    """Função objetivo do optuna"""
    return valida(cria_instancia_HGradientBoosting(trial), X, y, grupos, n_folds)

def funcao_objetivo_NB(trial, X, y, grupos, n_folds):
    """Função objetivo do optuna"""
    return valida(cria_instancia_NB(trial), X, y, grupos, n_folds)

def funcao_objetivo_Stacking(trial, X, y, grupos, n_folds):
    """Função objetivo do optuna"""
    return valida(cria_instancia_Stacking(trial), X, y, grupos, n_folds)

############################## Creat studies ###########################

def cria_estudo(nome):
    """Cria (ou carrega) o estudo NOVO, com o sufixo, em um banco próprio."""
    return create_study(
        direction="maximize",
        study_name=nome + SUFIXO,
        storage=f"sqlite:///{nome}{SUFIXO}.db",
        load_if_exists=True,
        sampler=GPSampler(),
    )

# Nomes dos estudos ANTIGOS (somente leitura, usados para tirar as sementes)
NOME_ESTUDO_kNN = "YSO_classifier-kNN"
NOME_ESTUDO_tree = "YSO_classifier-Tree"
NOME_ESTUDO_forest = "YSO_classifier-Forest"
NOME_ESTUDO_kNN_ensemble = "YSO_classifier-kNN_ensemble"
NOME_ESTUDO_Hist_Gradient_Boosting = "YSO_classifier-Hist_Gradient_Boosting"
NOME_ESTUDO_Naive_Bayes = "YSO_classifier-Naive_Bayes"
NOME_ESTUDO_Stacking = "YSO_classifier-Stacking-LinearSVC_Forest_NB-LogReg-Multilayer_or_not"

objeto_de_estudo_kNN = cria_estudo(NOME_ESTUDO_kNN)
objeto_de_estudo_tree = cria_estudo(NOME_ESTUDO_tree)
objeto_de_estudo_forest = cria_estudo(NOME_ESTUDO_forest)
objeto_de_estudo_kNN_ensemble = cria_estudo(NOME_ESTUDO_kNN_ensemble)
objeto_de_estudo_HGradientBoosting = cria_estudo(NOME_ESTUDO_Hist_Gradient_Boosting)
objeto_de_estudo_NB = cria_estudo(NOME_ESTUDO_Naive_Bayes)
objeto_de_estudo_Stacking = cria_estudo(NOME_ESTUDO_Stacking)

############################## Partial functions ###########################

NUM_FOLDS=10

def objetivo_parcial(funcao, n_folds):
    return partial(funcao, X=X_treino, y=y_treino, grupos=grupos, n_folds=n_folds)

funcao_objetivo_parcial_kNN = objetivo_parcial(funcao_objetivo_kNN, NUM_FOLDS)
funcao_objetivo_parcial_tree = objetivo_parcial(funcao_objetivo_tree, NUM_FOLDS)
funcao_objetivo_parcial_forest = objetivo_parcial(funcao_objetivo_forest, NUM_FOLDS)
funcao_objetivo_parcial_kNN_ensemble = objetivo_parcial(funcao_objetivo_kNN_ensemble, NUM_FOLDS)
funcao_objetivo_parcial_HGradientBoosting = objetivo_parcial(funcao_objetivo_HGradientBoosting, NUM_FOLDS)
funcao_objetivo_parcial_NB = objetivo_parcial(funcao_objetivo_NB, NUM_FOLDS)
funcao_objetivo_parcial_Stacking = objetivo_parcial(funcao_objetivo_Stacking, NUM_FOLDS//2)

############################## Forçando trials ###########################
# Os primeiros trials de cada estudo novo são os MELHORES trials do estudo antigo
# (reavaliados com os atributos e a validação novos). Se o estudo antigo não existir,
# usa-se uma configuração padrão. Os valores são ajustados às faixas de busca atuais.

K_SEMENTES = 2     # quantos dos melhores trials antigos reaproveitar por estudo

RENOMEIA = {"mins_samples_leaf": "min_samples_leaf"}   # nome antigo (com erro de digitação) -> atual

# Faixas numéricas atuais de cada espaço de busca (usadas para ajustar os valores antigos)
LIMITES_kNN = {"n_neighbors": (500, 5000), "p": (1.0, 2.0), "n_components": (10, 20)}
LIMITES_tree = {
    "max_depth": (3, 300), "min_samples_split": (0.001, 0.05), "min_samples_leaf": (0.001, 0.05),
    "min_weight_fraction_leaf": (0.0, 0.5), "max_leaf_nodes": (6, 500),
    "min_impurity_decrease": (0.0, 0.5), "n_components": (12, 20),
}
LIMITES_forest = {
    "n_estimators": (100, 500), "max_depth": (3, 500), "min_samples_split": (0.001, 0.05),
    "min_samples_leaf": (0.001, 0.05), "min_weight_fraction_leaf": (0.0, 0.5),
    "max_features": (0.2, 1.0), "max_leaf_nodes": (6, 500),
    "min_impurity_decrease": (0.0, 0.5), "n_components": (12, 20),
}
LIMITES_kNN_ensemble = {
    "n_neighbors": (500, 5000), "p": (1.0, 2.0), "n_estimators": (10, 500),
    "max_samples": (0.7, 1.0), "max_features": (0.2, 1.0), "n_components": (10, 15),
}
LIMITES_HGB = {
    "learning_rate": (0.01, 1.0), "max_iter": (100, 500), "max_leaf_nodes": (5, 150),
    "max_depth": (3, 300), "min_samples_leaf": (100, 1000), "l2_regularization": (0.0, 9.0),
    "max_features": (0.2, 1.0), "max_bins": (100, 255), "n_components": (10, 18),
}
LIMITES_Stacking = dict(LIMITES_forest, n_components=(10, 18))   # parte da floresta no Stacking

# Parâmetros da floresta dentro do Stacking
CHAVES_FLORESTA = [
    "n_estimators", "criterion", "max_depth", "min_samples_split", "min_samples_leaf",
    "min_weight_fraction_leaf", "max_features", "max_leaf_nodes", "min_impurity_decrease",
    "class_weight", "n_components",
]


def melhores_params(nome_antigo, k=K_SEMENTES):
    """Parâmetros dos k melhores trials de um estudo antigo ([] se ele não existir)."""
    arquivo = f"{nome_antigo}.db"
    if not os.path.exists(arquivo):     # evita criar um banco vazio sem querer
        return []
    try:
        estudo = optuna.load_study(study_name=nome_antigo, storage=f"sqlite:///{arquivo}")
    except KeyError:
        return []
    concluidos = [t for t in estudo.trials if t.state == TrialState.COMPLETE and t.value is not None]
    concluidos.sort(key=lambda t: t.value, reverse=True)
    return [dict(t.params) for t in concluidos[:k]]


def ajusta(params, limites):
    """Renomeia parâmetros antigos e leva os valores numéricos para dentro das faixas atuais."""
    novo = {}
    for nome, valor in params.items():
        nome = RENOMEIA.get(nome, nome)
        if nome in limites:
            inferior, superior = limites[nome]
            valor = min(max(valor, inferior), superior)
            if isinstance(inferior, int):
                valor = int(round(valor))
        novo[nome] = valor
    return novo


def semeia(estudo, nome_antigo, limites, padroes):
    """Coloca na fila os melhores trials antigos (ou os padrões, se não houver estudo antigo)."""
    sementes = [ajusta(p, limites) for p in melhores_params(nome_antigo)]
    for params in (sementes or padroes):
        estudo.enqueue_trial(params, skip_if_exists=True)
    print(f"{estudo.study_name}: {len(sementes)} semente(s) do estudo antigo"
          f"{'' if sementes else ' (usando configuração padrão)'}")
    return sementes


# kNN
semeia(objeto_de_estudo_kNN, NOME_ESTUDO_kNN, LIMITES_kNN, [
    {"n_neighbors": 500, "weights": "uniform", "p": 2.0},
])

# Tree
semeia(objeto_de_estudo_tree, NOME_ESTUDO_tree, LIMITES_tree, [{
    "criterion": "gini", "max_depth": 100, "min_samples_split": 0.001,
    "min_samples_leaf": 0.001, "min_weight_fraction_leaf": 0.0,
    "max_leaf_nodes": 200, "min_impurity_decrease": 0.0,
}])

# Forest
PADRAO_FLORESTA = {
    "n_estimators": 100, "criterion": "gini", "max_depth": 100, "min_samples_split": 0.001,
    "min_samples_leaf": 0.001, "min_weight_fraction_leaf": 0.0, "max_features": 0.3,
    "max_leaf_nodes": 200, "min_impurity_decrease": 0.0, "class_weight": "balanced",
}
semeia(objeto_de_estudo_forest, NOME_ESTUDO_forest, LIMITES_forest, [
    PADRAO_FLORESTA, dict(PADRAO_FLORESTA, class_weight="balanced_subsample"),
])

# kNN_ensemble
semeia(objeto_de_estudo_kNN_ensemble, NOME_ESTUDO_kNN_ensemble, LIMITES_kNN_ensemble, [
    {"n_neighbors": 500, "weights": "uniform", "p": 2.0,
     "n_estimators": 10, "max_samples": 1.0, "max_features": 1.0},
    {"n_neighbors": 500, "weights": "uniform", "p": 2.0,
     "n_estimators": 10, "max_samples": 1.0, "max_features": 0.3},
])

# HistGradientBoosting
semeia(objeto_de_estudo_HGradientBoosting, NOME_ESTUDO_Hist_Gradient_Boosting, LIMITES_HGB, [
    {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31, "max_depth": 100,
     "min_samples_leaf": 100, "l2_regularization": 0.0, "max_features": 1.0, "max_bins": 255},
    {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31, "max_depth": 100,
     "min_samples_leaf": 100, "l2_regularization": 0.0, "max_features": 0.3, "max_bins": 255},
])

# Naive Bayes (o estudo antigo usava outros atributos: não vale reaproveitar)
for suavizacao in (1e-9, 1e-6, 1e-3):
    objeto_de_estudo_NB.enqueue_trial({"var_smoothing": suavizacao}, skip_if_exists=True)

# Stacking: a floresta interna parte dos melhores parâmetros do estudo da FLORESTA,
# combinados com cada opção de multicamada e passthrough.
sementes_floresta = [
    {k: v for k, v in ajusta(p, LIMITES_Stacking).items() if k in CHAVES_FLORESTA}
    for p in melhores_params(NOME_ESTUDO_forest)
] or [dict(PADRAO_FLORESTA, n_components=14)]

for floresta in sementes_floresta:
    for multicamada in (False, True):
        for passthrough in (False, True):
            params = {
                "n_components": 14,
                # LinearSVC
                "penalty": "l2", "tol": 1e-4, "C": 1.0,
                # Naive Bayes
                "var_smoothing": 1e-9,
                # Estimador final
                "C_final": 1.0,
                **floresta,
                "multilayer": multicamada,
                "passthrough": passthrough,
            }
            if multicamada:
                params["C_camada2"] = 1.0
            objeto_de_estudo_Stacking.enqueue_trial(params, skip_if_exists=True)

############################## Otimizando ###########################

NUM_TRIALS = 200

objeto_de_estudo_NB.optimize(
    funcao_objetivo_parcial_NB, n_trials=NUM_TRIALS
)

objeto_de_estudo_tree.optimize(
    funcao_objetivo_parcial_tree, n_trials=NUM_TRIALS//2
)

objeto_de_estudo_forest.optimize(
    funcao_objetivo_parcial_forest, n_trials=NUM_TRIALS
)

objeto_de_estudo_HGradientBoosting.optimize(
    funcao_objetivo_parcial_HGradientBoosting, n_trials=NUM_TRIALS
)

objeto_de_estudo_kNN.optimize(
    funcao_objetivo_parcial_kNN, n_trials=NUM_TRIALS//2
)

objeto_de_estudo_Stacking.optimize(
    funcao_objetivo_parcial_Stacking, n_trials=NUM_TRIALS
)

objeto_de_estudo_kNN_ensemble.optimize(
    funcao_objetivo_parcial_kNN_ensemble, n_trials=NUM_TRIALS
)

############################## Melhores trials ###########################

for nome, estudo in [
    ("k-NN", objeto_de_estudo_kNN),
    ("Árvore", objeto_de_estudo_tree),
    ("Floresta", objeto_de_estudo_forest),
    ("k-NN com bagging", objeto_de_estudo_kNN_ensemble),
    ("HistGradientBoosting", objeto_de_estudo_HGradientBoosting),
    ("Naive Bayes", objeto_de_estudo_NB),
    ("Stacking", objeto_de_estudo_Stacking),
]:
    melhor = estudo.best_trial
    print(f"[{nome}] Número do melhor trial: {melhor.number} | ROC-AUC: {melhor.value:.4f}")
    print(f"[{nome}] Parâmetros do melhor trial: {melhor.params}")