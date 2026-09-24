"""Gradient Boosting (ensemble de árvores) sobre os mesmos dados e divisão da
árvore única, para comparação sob critérios equivalentes (ver docs/DECISOES.md, item 6)."""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from . import config


def treinar_gradient_boosting(x_treino: pd.DataFrame, y_treino: pd.Series) -> GradientBoostingClassifier:
    modelo = GradientBoostingClassifier(
        n_estimators=config.GB_N_ESTIMATORS,
        learning_rate=config.GB_LEARNING_RATE,
        max_depth=config.GB_MAX_DEPTH,
        random_state=config.SEMENTE_PRINCIPAL,
    )
    modelo.fit(x_treino, y_treino)
    return modelo


def importancia_das_variaveis(modelo: GradientBoostingClassifier, nomes_colunas: list) -> pd.Series:
    return pd.Series(modelo.feature_importances_, index=nomes_colunas).sort_values(ascending=False)
