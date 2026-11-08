############################## Importações ##############################
# Bibliotecas gerais
import os                   # Usado para checar se os bancos dos estudos antigos existem.
import pandas as pd         # Usado para carregar os arquivos.
import numpy as np          # Usado para trabalhar com arrays.

# Modelagem
    # Algoritmos
from sklearn.pipeline import make_pipeline              # Cria pipeline.
from sklearn.svm import LinearSVC                       # SVM Linear SVC.
from sklearn.linear_model import LogisticRegression     # Regressão logística (estimador final do Stacking).
from sklearn.naive_bayes import GaussianNB              # Algoritmo Naive-Bayes Gaussiano.
        # Algoritmos de comitê
from sklearn.ensemble import (
    RandomForestClassifier,             # Floresta Aleatória classificadora.
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

# Paralelizacao: usa o que o Slurm liberou, nao o total do no
N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
N_JOBS_FOLDS = min(5, N_CPUS)                      # folds externos em paralelo
N_JOBS_FLORESTA = max(1, N_CPUS // N_JOBS_FOLDS)   # com 12 CPUs: 2 por floresta

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
y_treino = df_treino[TARGET].to_numpy(dtype="object")     # array 1-D (um DataFrame (n,1) zera todas as metricas)

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
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 3, 500, log=True),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.05, log=True),
        "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.001, 0.05, log=True),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0, step=0.05),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 6, 500, log=True),
        # Argumentos fixos
        "n_estimators": 200,
        "min_weight_fraction_leaf": 0.0,
        "min_impurity_decrease": 0.0,
        "class_weight": "balanced_subsample",
        "n_jobs": N_JOBS_FLORESTA,
        "bootstrap": True,
        "random_state": RANDOM_SEED,
    }

    parametros_NB = {
        # Espaço de busca
        "var_smoothing": 1e-9
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
        "n_jobs": 1,
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
            n_jobs=1,
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
        cv=cv,
	n_jobs=N_JOBS_FOLDS
    )

    return metricas.mean()

def funcao_objetivo_forest(trial, X, y, grupos, n_folds):
    """Função objetivo do optuna"""
    return valida(cria_instancia_forest(trial), X, y, grupos, n_folds)

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
NOME_ESTUDO_forest = "YSO_classifier-Forest"
NOME_ESTUDO_Stacking = "YSO_classifier-Stacking-LinearSVC_Forest_NB-LogReg-Multilayer_or_not"

objeto_de_estudo_forest = cria_estudo(NOME_ESTUDO_forest)
objeto_de_estudo_Stacking = cria_estudo(NOME_ESTUDO_Stacking)

############################## Partial functions ###########################

NUM_FOLDS=10

def objetivo_parcial(funcao, n_folds):
    return partial(funcao, X=X_treino, y=y_treino, grupos=grupos, n_folds=n_folds)

funcao_objetivo_parcial_forest = objetivo_parcial(funcao_objetivo_forest, NUM_FOLDS)
funcao_objetivo_parcial_Stacking = objetivo_parcial(funcao_objetivo_Stacking, NUM_FOLDS//2)

############################## Forçando trials ###########################
# Os primeiros trials de cada estudo novo são os MELHORES trials do estudo antigo
# (reavaliados com os atributos e a validação novos). Se o estudo antigo não existir,
# usa-se uma configuração padrão. Os valores são ajustados às faixas de busca atuais.

K_SEMENTES = 2     # quantos dos melhores trials antigos reaproveitar por estudo

RENOMEIA = {"mins_samples_leaf": "min_samples_leaf"}   # nome antigo (com erro de digitação) -> atual

# Faixas numéricas atuais de cada espaço de busca (usadas para ajustar os valores antigos)
LIMITES_forest = {
    "n_estimators": (100, 500), "max_depth": (3, 500), "min_samples_split": (0.001, 0.05),
    "min_samples_leaf": (0.001, 0.05), "min_weight_fraction_leaf": (0.0, 0.5),
    "max_features": (0.2, 1.0), "max_leaf_nodes": (6, 500),
    "min_impurity_decrease": (0.0, 0.5), "n_components": (12, 20),
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


# Forest
PADRAO_FLORESTA = {
    "n_estimators": 100, "criterion": "gini", "max_depth": 100, "min_samples_split": 0.001,
    "min_samples_leaf": 0.001, "min_weight_fraction_leaf": 0.0, "max_features": 0.3,
    "max_leaf_nodes": 200, "min_impurity_decrease": 0.0, "class_weight": "balanced",
}
semeia(objeto_de_estudo_forest, NOME_ESTUDO_forest, LIMITES_forest, [
    PADRAO_FLORESTA, dict(PADRAO_FLORESTA, class_weight="balanced_subsample"),
])

# Stacking: a floresta interna parte dos melhores parâmetros do estudo da FLORESTA,
# combinados com cada opção de multicamada e passthrough.
sementes_floresta = [
    {k: v for k, v in ajusta(p, LIMITES_Stacking).items() if k in CHAVES_FLORESTA}
    for p in melhores_params(NOME_ESTUDO_forest + SUFIXO)
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

NUM_TRIALS = 200 - 37	# Ja foram feitos 37 trials

objeto_de_estudo_Stacking.optimize(
    funcao_objetivo_parcial_Stacking, n_trials=NUM_TRIALS
)

############################## Melhores trials ###########################

for nome, estudo in [
    ("Floresta", objeto_de_estudo_forest),
    ("Stacking", objeto_de_estudo_Stacking),
]:
    melhor = estudo.best_trial
    print(f"[{nome}] Número do melhor trial: {melhor.number} | ROC-AUC: {melhor.value:.4f}")
    print(f"[{nome}] Parâmetros do melhor trial: {melhor.params}")