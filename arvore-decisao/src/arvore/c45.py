"""Wrapper fino em torno da chefboost (C4.5 real — ver docs/DECISOES.md, item 1).

Isola dois workarounds de ambiente que a chefboost precisa para funcionar com
pandas moderno fora de notebook:
1. Converte só as colunas de texto (não as numéricas) para dtype `object`
   puro antes de treinar — chefboost quebra com `string`/`category`.
2. Garante o cwd em `sys.path`, pois ela importa dinamicamente as regras
   geradas em `outputs/rules/rules.py` via `importlib`.
"""

import os
import sys
from dataclasses import dataclass, field

import pandas as pd

from . import config

COLUNA_ALVO_CHEFBOOST = "Decision"


@dataclass
class ArvoreC45:
    modelo: object
    colunas_atributos: list
    profundidade_maxima: int | None
    caminho_regras: str = field(default="outputs/rules/rules.py")

    def prever(self, df: pd.DataFrame):
        from chefboost import Chefboost as chef

        linhas = df[self.colunas_atributos].values.tolist()
        return [chef.predict(self.modelo, linha) for linha in linhas]

    def texto_das_regras(self) -> str:
        with open(self.caminho_regras, encoding="utf-8") as arquivo:
            return arquivo.read()


def _garantir_ambiente_chefboost() -> None:
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def treinar_c45(
    df_treino: pd.DataFrame,
    profundidade_maxima: int | None = None,
    silencioso: bool = True,
) -> ArvoreC45:
    """`df_treino` deve já estar no formato de `dados.preparar_para_c45`."""
    _garantir_ambiente_chefboost()
    from chefboost import Chefboost as chef

    config_chefboost = {"algorithm": "C4.5", "enableParallelism": False}
    if profundidade_maxima is not None:
        config_chefboost["max_depth"] = profundidade_maxima

    kwargs = {"config": config_chefboost, "target_label": COLUNA_ALVO_CHEFBOOST}
    if silencioso:
        kwargs["silent"] = True

    modelo = chef.fit(df_treino.reset_index(drop=True).copy(), **kwargs)
    colunas_atributos = [c for c in df_treino.columns if c != COLUNA_ALVO_CHEFBOOST]
    return ArvoreC45(modelo=modelo, colunas_atributos=colunas_atributos, profundidade_maxima=profundidade_maxima)


def avaliar_c45(arvore: ArvoreC45, df: pd.DataFrame) -> float:
    previsoes = arvore.prever(df)
    reais = df[COLUNA_ALVO_CHEFBOOST].tolist()
    acertos = sum(1 for p, r in zip(previsoes, reais) if p == r)
    return acertos / len(reais)
