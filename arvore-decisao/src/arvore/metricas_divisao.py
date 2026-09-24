"""Métricas usadas para escolher divisões em árvores de decisão: entropia,
ganho de informação, split information e razão de ganho (gain ratio)."""

import math

import pandas as pd


def entropia(rotulos: pd.Series) -> float:
    if len(rotulos) == 0:
        return 0.0
    contagens = rotulos.value_counts()
    probabilidades = contagens / len(rotulos)
    return -sum(p * math.log2(p) for p in probabilidades if p > 0)


def ganho_informacao(df: pd.DataFrame, atributo: str, alvo: str) -> float:
    entropia_antes = entropia(df[alvo])
    n = len(df)
    entropia_depois = 0.0
    for _, grupo in df.groupby(atributo, observed=True):
        peso = len(grupo) / n
        entropia_depois += peso * entropia(grupo[alvo])
    return entropia_antes - entropia_depois


def split_information(df: pd.DataFrame, atributo: str) -> float:
    n = len(df)
    contagens = df[atributo].value_counts()
    probabilidades = contagens / n
    return -sum(p * math.log2(p) for p in probabilidades if p > 0)


def razao_de_ganho(df: pd.DataFrame, atributo: str, alvo: str) -> float:
    si = split_information(df, atributo)
    if si == 0:
        return 0.0
    return ganho_informacao(df, atributo, alvo) / si


def melhor_atributo_por_razao_de_ganho(df: pd.DataFrame, atributos: list, alvo: str) -> dict:
    """Ranking de atributos por razão de ganho, do maior para o menor."""
    resultado = {}
    for atributo in atributos:
        ganho = ganho_informacao(df, atributo, alvo)
        si = split_information(df, atributo)
        razao = ganho / si if si > 0 else 0.0
        resultado[atributo] = {"ganho": ganho, "split_info": si, "razao_ganho": razao}
    return dict(sorted(resultado.items(), key=lambda item: item[1]["razao_ganho"], reverse=True))
