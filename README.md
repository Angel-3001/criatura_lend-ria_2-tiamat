# Tiamat — Classificação de estágios evolutivos de YSOs

> **Estado do projeto:** em desenvolvimento — versão de acompanhamento em 9 de novembro de 2026.  
> **Objetivo:** comparar classificadores de aprendizado de máquina para distinguir quatro classes de *Young Stellar Objects* (YSOs) a partir de dados do catálogo SPICY.

## 1. Visão geral

O projeto investiga o uso de métodos de aprendizado de máquina para classificar objetos estelares jovens (*Young Stellar Objects*, YSOs) de acordo dados de fotometria de infravermelho. A análise concentra-se em quatro classes — **Class I**, **Flat Spectrum (FS)**, **Class II** e **Class III** — e compara modelos individuais e métodos de comitê.

A motivação é avaliar se atributos astrométricos, ambientais e principalmente fotométricos permitem distinguir classes cujas distribuições podem apresentar sobreposição. Também se pretende estudar o compromisso entre qualidade preditiva e custo computacional, sobretudo em Florestas Aleatórias e modelos de *Stacking*.

**Escopo importante:** o material atual trabalha com a classificação *entre quatro classes de YSOs*. Não é, nesta etapa, um classificador binário de YSO versus não-YSO.

## 2. Dados e classes

O conjunto de dados vem do catálogo **SPICY**, que reúne informações de diferentes levantamentos astronômicos e candidatos a YSO. Após as etapas de limpeza descritas no notebook, foram registrados **101.646 objetos** distribuídos da seguinte forma:

| Classe | Quantidade após limpeza | Interpretação usada no projeto |
|---|---:|---|
| Class I | 15.253 | Estágio jovem, associado à fase de protoestrela e forte envoltório de material. |
| FS | 23.068 | Espectro aproximadamente plano (*Flat Spectrum*), classe intermediária entre I e II. |
| Class II | 58.114 | Objeto com disco circunstelar, em estágio mais evoluído que Class I/FS. |
| Class III | 5.211 | Estágio mais evoluído entre as quatro classes consideradas, com emissão infravermelha menos acentuada. |

A distribuição é desbalanceada: Class II representa a maior parcela, enquanto Class III é a classe minoritária. Por isso, acurácia isolada não é suficiente para avaliar os modelos.

### Atributos utilizados

O script de otimização usa 22 atributos preditores:

- Coordenadas equatoriais e galácticas: `ra`, `dec`, `l`, `b`;
- Ambiente infravermelho: `env`;
- Magnitudes e erros fotométricos Spitzer/IRAC: `mag3_6`, `e_mag3_6`, `mag4_5`, `e_mag4_5`, `mag5_8`, `e_mag5_8`, `mag8_0`, `e_mag8_0`;
- Indicador de possível contaminação: `csf`;
- Número de detecções: `m3_6`, `m4_5`, `m5_8`, `m8_0`;
- Cores construídas a partir das diferenças entre magnitudes: `c_mag36_45`, `c_mag45_58`, `c_mag58_80`, `c_mag36_80`.

A coluna `class` é o alvo. A coluna `group` é mantida para a separação dos dados por grupos espaciais e **não deve entrar como atributo do classificador**.

### Limpeza e engenharia de atributos

As decisões documentadas no notebook incluem:

1. Excluir registros sem os atributos necessários, registros classificados como `uncertain` e ambientes marcados como `Outliers`.
2. Remover atributos associados a classificações ou pontuações anteriores — por exemplo `p1`, `p2`, `p3` e índices espectrais usados na classificação original — para reduzir o risco de vazamento de informação (*data leakage*).
3. Remover `silicate` e `pah` na seleção atual de atributos.
4. Transformar a informação de ambiente em categorias ordinais `EnvI`, `EnvII` e `EnvIII`.
5. Construir quatro atributos de cor pela diferença entre magnitudes de bandas infravermelhas.
6. Preservar `group` exclusivamente para separar regiões/grupos entre treino, validação e teste. Para registros sem grupo, o notebook cria identificadores individuais, evitando que todos sejam tratados como um único grupo.

As cores foram incluídas porque suas distribuições parecem separar melhor as classes do que as magnitudes isoladas e são especialmente pertinentes à hipótese de uso de Gaussian Naive Bayes. Contudo, como a versão nova também alterou o protocolo de validação, a comparação entre versões antigas e novas não isola o efeito das cores: não se deve atribuir todo ganho observado exclusivamente à engenharia de atributos.

## 3. Estratégia de validação

O projeto usa `StratifiedGroupKFold`, com a intenção de manter a proporção das classes e evitar que observações pertencentes ao mesmo grupo espacial apareçam simultaneamente em treino e validação. Essa precaução é importante porque objetos próximos podem compartilhar condições ambientais ou características observacionais.

O arquivo de treino atualmente fornecido, `YSOs-Dados_treino-Ordinal_encoder.csv`, contém **91.384 linhas**, com a seguinte distribuição:

| Classe | Treino |
|---|---:|
| Class I | 13.714 |
| FS | 20.644 |
| Class II | 52.303 |
| Class III | 4.723 |

O notebook registra uma divisão inicial de aproximadamente 90% para treino e 10% para teste por grupos: 91.384 observações de treino e 10.262 de teste. Entretanto, o CSV de teste não está entre os arquivos entregues nesta versão do projeto. Ele deve ser salvo e preservado separadamente para a avaliação final.

### Métrica principal

A otimização usa **ROC-AUC multiclasse OvR (*one-vs-rest*), com média ponderada** pela frequência das classes (`average="weighted"`). Essa métrica avalia a capacidade de ranqueamento das probabilidades estimadas para cada classe.

Ela deve ser acompanhada de métricas que exponham o comportamento das classes minoritárias:

- ROC-AUC macro OvR e ROC-AUC por classe;
- F1 macro;
- *Balanced accuracy*;
- Precisão e *recall* por classe;
- Matriz de confusão e relatório de classificação;
- Média e desvio-padrão entre folds, além do tempo de treino e de predição.

O baseline com `DummyClassifier` obteve **ROC-AUC ponderada de 0,500**, acurácia de **0,57** e *recall* macro de **0,25**. Ele prevê essencialmente a classe majoritária: o *recall* de Class I, Class III e FS foi zero. Isso mostra por que uma acurácia aparentemente razoável pode ocultar a incapacidade de identificar classes minoritárias.

## 4. Modelos investigados

O projeto compara:

- **k-NN:** classificador por vizinhos mais próximos, com PCA e normalização;
- **Árvore de decisão:** modelo individual baseado em partições sucessivas;
- **Floresta Aleatória:** conjunto de árvores treinadas com amostras e subconjuntos de atributos;
- **Bagging de k-NN:** conjunto de classificadores k-NN;
- **HistGradientBoosting:** método de *gradient boosting* baseado em histogramas;
- **Gaussian Naive Bayes:** usa somente magnitudes, erros e cores fotométricas, com transformação quantílica;
- **Stacking:** combina `LinearSVC`, Floresta Aleatória e Gaussian Naive Bayes; a camada final usa Regressão Logística. Também foi explorada uma variante multicamada.

No *Stacking*, o Naive Bayes recebe apenas os atributos fotométricos e de cor (`ATRIBUTOS_NB`), enquanto os outros estimadores recebem a lista completa de atributos. O `LinearSVC` foi envolvido por `CalibratedClassifierCV` para produzir probabilidades, e a camada final recebe as saídas dos estimadores-base. O parâmetro `passthrough` determina se os atributos originais também são entregues à camada final.

Os hiperparâmetros foram explorados com **Optuna** e os estudos são persistidos em bancos SQLite. A versão revisada do script aproveita os melhores hiperparâmetros encontrados em estudos anteriores como pontos de partida para novos estudos com o sufixo `_v2`.

## 5. Resultados preliminares de otimização

A tabela resume o melhor valor de ROC-AUC ponderada encontrado nos bancos `_v2` fornecidos. Para o k-NN, o valor foi recuperado do notebook de análise, pois o banco `_v2` correspondente não foi anexado.

| Modelo | Melhor ROC-AUC ponderada | Trials completos | Folds externos usados na otimização | Mediana do tempo por trial |
|---|---:|---:|---:|---:|
| Árvore de decisão | 0,891081 | 100 | 10 | 21,9 s |
| Floresta Aleatória | 0,911715 | 200 | 10 | 171,8 s |
| HistGradientBoosting | 0,930052 | 200 | 10 | 46,7 s |
| Gaussian Naive Bayes | 0,883177 | 200 | 10 | 11,4 s |
| Stacking multicamada | **0,933560** | 115 (mais 4 em execução no banco) | **5** | **663,4 s** |
| k-NN (v1) | 0,883715 | 124 | 10 | Não foi possível determinar |

**Como interpretar a tabela:** são os melhores valores da função objetivo durante a busca de hiperparâmetros, não resultados de um teste independente. O Stacking foi otimizado com cinco folds externos, enquanto os demais modelos da tabela usaram dez. Portanto, essa tabela é um retrato do progresso da otimização, não uma comparação final perfeitamente controlada.

**Obs.:** Só foi possível analisar a primeira otimização do k-NN, pois a segunda não pode ser finalizada em tempo hábil.

### Leitura inicial dos resultados

1. A Floresta Aleatória melhorou em relação à configuração antiga: o melhor valor passou de aproximadamente 0,8947 para 0,9117. Ainda assim, a versão nova alterou atributos e validação, então essa diferença não pode ser atribuída a uma única mudança. Porém, ao analisar outras métricas, como *macro F-1*, percebe-se que esse não foi um desempenho real, precisando levar em conta, portanto, outras métricas para a avaliação.
2. O HistGradientBoosting atingiu 0,9301 com mediana de 46,7 s por trial, valor substancialmente superior ao da floresta e com custo de otimização menor. Além disso, ao usar um conjunto de validação, observou-se métricas extremamente condizentes com esse resultado, diferentemente da árvore.
3. O Stacking chegou a 0,9336, aproximadamente **0,0035** acima do HistGradientBoosting (cerca de **0,35 ponto percentual**). Em contrapartida, a mediana de duração por trial foi de aproximadamente 663 s, cerca de **14 vezes** a mediana do HGB. Além disso, não foi possível usar um conjunto de validação para analisar o desempenho desse algoritmo em tempo hábil, tornando difícil estimar seu desempenho real. Outra coisa importante é que, após essas análises, concluíu-se que, provavelmente, o desempenho e custo computacional desse algoritmo melhorassem ao usar HistGradientBoosting ao invés de RandomForest.
4. A melhor configuração de floresta atingiu o limite superior da busca em `n_estimators=500` e `max_leaf_nodes=500`. Isso indica que a região superior ainda merece investigação, mas não demonstra que aumentar esses valores seja vantajoso: o acréscimo de custo observado e análise de outras métricas, apontam, inclusive, que aumentar a complexidade da floresta pode gerar mais *overfitting* ou ter pouco ganho computacional.
5. O melhor Stacking registrado é multicamada e usa `passthrough=True`; a floresta da primeira camada também está configurada com 500 árvores, o que explica o custo computacional elevado.

**Conclusão provisória:** o HistGradientBoosting é, por enquanto, o candidato mais atraente em desempenho por custo computacional. O Stacking é o melhor valor observado na busca, mas seu ganho é pequeno, seu custo é muito maior e a comparação externa usa número de folds diferente. Não há base suficiente para declará-lo vencedor definitivo.

## 10. Ambiente e execução

O código fornecido utiliza Python, NumPy, pandas, scikit-learn e Optuna, além de Matplotlib, Seaborn e SciPy nos notebooks de análise. As versões exatas não estão registradas nos arquivos recebidos; portanto, devem ser verificadas no ambiente original antes de fixar dependências.

Exemplo de instalação inicial:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# No Windows: .venv\Scripts\activate

python -m pip install numpy pandas scikit-learn optuna matplotlib seaborn scipy
```

Antes de executar a otimização, confirme os caminhos dos CSVs, os nomes dos bancos SQLite e a configuração de CPU. Os scripts usam `load_if_exists=True`, então a execução pode continuar um estudo existente.