############################## Importações ##############################
# Bibliotecas gerais
import os                   # Usado para checar se os bancos dos estudos antigos existem.
import pandas as pd         # Usado para carregar os arquivos.
import numpy as np          # Usado para trabalhar com arrays.

# Modelagem
    # Algoritmos
from sklearn.pipeline import make_pipeline              # Cria pipeline.
from sklearn.neighbors import KNeighborsClassifier      # Classificador k-NN.
    # Outras coisas mais
from sklearn.model_selection import (
    cross_val_score,                                        # Validação cruzada.
    StratifiedKFold,                                        # Cria k-Folds estratificados.
    StratifiedGroupKFold                                    # k-Folds estratificados e por grupos.
)
from sklearn.preprocessing import StandardScaler            # Normalizador padrão.
from sklearn.decomposition import PCA                       # Redução de dimensionalidade
                                                            #   por PCA.
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

objeto_de_estudo_kNN = cria_estudo(NOME_ESTUDO_kNN)

############################## Partial functions ###########################

NUM_FOLDS=10

def objetivo_parcial(funcao, n_folds):
    return partial(funcao, X=X_treino, y=y_treino, grupos=grupos, n_folds=n_folds)

funcao_objetivo_parcial_kNN = objetivo_parcial(funcao_objetivo_kNN, NUM_FOLDS)

############################## Forçando trials ###########################
# Os primeiros trials de cada estudo novo são os MELHORES trials do estudo antigo
# (reavaliados com os atributos e a validação novos). Se o estudo antigo não existir,
# usa-se uma configuração padrão. Os valores são ajustados às faixas de busca atuais.

K_SEMENTES = 2     # quantos dos melhores trials antigos reaproveitar por estudo

RENOMEIA = {"mins_samples_leaf": "min_samples_leaf"}   # nome antigo (com erro de digitação) -> atual

# Faixas numéricas atuais de cada espaço de busca (usadas para ajustar os valores antigos)
LIMITES_kNN = {"n_neighbors": (500, 5000), "p": (1.0, 2.0), "n_components": (10, 20)}

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

############################## Otimizando ###########################

NUM_TRIALS = 200

objeto_de_estudo_kNN.optimize(
    funcao_objetivo_parcial_kNN, n_trials=NUM_TRIALS//2
)

############################## Melhores trials ###########################

for nome, estudo in [
    ("k-NN", objeto_de_estudo_kNN)
]:
    melhor = estudo.best_trial
    print(f"[{nome}] Número do melhor trial: {melhor.number} | ROC-AUC: {melhor.value:.4f}")
    print(f"[{nome}] Parâmetros do melhor trial: {melhor.params}")