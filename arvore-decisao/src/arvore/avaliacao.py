"""Linhas de base, matriz de confusão, estabilidade e exemplos de acertos/erros —
contexto além da acurácia bruta (ver docs/DECISOES.md, item 7)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from . import config


def baseline_classe_majoritaria(y: pd.Series) -> float:
    return y.value_counts(normalize=True).max()


def baseline_um_corte(x_treino: pd.DataFrame, y_treino: pd.Series, semente: int = config.SEMENTE_PRINCIPAL) -> float:
    arvore = DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=semente)
    dobras_scores = cross_val_score(arvore, x_treino, y_treino, cv=config.DOBRAS_VALIDACAO_CRUZADA)
    return dobras_scores.mean()


def matriz_de_confusao_legivel(y_real: pd.Series, y_previsto, rotulo_positivo: str = config.ROTULO_POSITIVO) -> pd.DataFrame:
    rotulos = sorted(set(y_real) | set(y_previsto))
    matriz = confusion_matrix(y_real, y_previsto, labels=rotulos)
    return pd.DataFrame(matriz, index=[f"Real: {r}" for r in rotulos], columns=[f"Previsto: {r}" for r in rotulos])


def taxa_de_erro_por_tipo(y_real: pd.Series, y_previsto, rotulo_positivo: str = config.ROTULO_POSITIVO) -> dict:
    """Separa os dois tipos de erro: aprovar um mau pagador custa mais caro
    que recusar um bom pagador (ver docs/DECISOES.md, item 7)."""
    y_real = pd.Series(y_real).reset_index(drop=True)
    y_previsto = pd.Series(y_previsto).reset_index(drop=True)
    positivos_reais = y_real == rotulo_positivo
    negativos_reais = ~positivos_reais

    falsos_negativos = ((y_previsto != rotulo_positivo) & positivos_reais).sum()
    falsos_positivos = ((y_previsto == rotulo_positivo) & negativos_reais).sum()

    return {
        "falsos_negativos_mau_pagador_aprovado": int(falsos_negativos),
        "falsos_positivos_bom_pagador_recusado": int(falsos_positivos),
        "total_maus_pagadores_reais": int(positivos_reais.sum()),
        "total_bons_pagadores_reais": int(negativos_reais.sum()),
    }


@dataclass
class ResultadoEstabilidade:
    sementes: list
    acuracias: list
    atributos_raiz: list

    @property
    def media_acuracia(self) -> float:
        return float(np.mean(self.acuracias))

    @property
    def desvio_padrao_acuracia(self) -> float:
        return float(np.std(self.acuracias))

    @property
    def raiz_mais_frequente(self) -> str:
        return pd.Series(self.atributos_raiz).value_counts().idxmax()

    @property
    def proporcao_raiz_estavel(self) -> float:
        return pd.Series(self.atributos_raiz).value_counts(normalize=True).max()


def experimento_estabilidade_cart(
    df_original: pd.DataFrame,
    df_cart: pd.DataFrame,
    profundidade_maxima,
    min_amostras_folha: int,
    sementes: list = config.SEMENTES_ESTABILIDADE,
) -> ResultadoEstabilidade:
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier

    from .dados import dividir_treino_teste

    acuracias, raizes = [], []
    for semente in sementes:
        treino, teste = dividir_treino_teste(df_original, config.COLUNA_ALVO, semente=semente)
        treino_cart, teste_cart = df_cart.loc[treino.index], df_cart.loc[teste.index]
        x_treino, y_treino = treino_cart.drop(columns=[config.COLUNA_ALVO]), treino_cart[config.COLUNA_ALVO]
        x_teste, y_teste = teste_cart.drop(columns=[config.COLUNA_ALVO]), teste_cart[config.COLUNA_ALVO]

        arvore = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=profundidade_maxima,
            min_samples_leaf=min_amostras_folha,
            random_state=semente,
        )
        arvore.fit(x_treino, y_treino)
        acuracias.append(accuracy_score(y_teste, arvore.predict(x_teste)))
        raizes.append(x_treino.columns[arvore.tree_.feature[0]])

    return ResultadoEstabilidade(sementes=list(sementes), acuracias=acuracias, atributos_raiz=raizes)


def selecionar_exemplos(
    x_teste: pd.DataFrame, y_teste: pd.Series, y_previsto, n_exemplos: int = 3, acertos: bool = True
) -> pd.DataFrame:
    y_previsto = pd.Series(y_previsto, index=x_teste.index)
    mascara_acerto = y_previsto == y_teste
    mascara = mascara_acerto if acertos else ~mascara_acerto
    indices = x_teste[mascara].index[:n_exemplos]
    exemplos = x_teste.loc[indices].copy()
    exemplos["classe_real"] = y_teste.loc[indices]
    exemplos["classe_prevista"] = y_previsto.loc[indices]
    return exemplos
