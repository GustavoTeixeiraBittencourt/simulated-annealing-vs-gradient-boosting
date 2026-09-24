"""Carga, checagem e preparação do dataset German Credit (`credit-g`, OpenML)."""

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from . import config


@dataclass
class ChecagemDataset:
    n_linhas: int
    n_colunas: int
    colunas_categoricas: list
    colunas_numericas: list
    n_valores_faltantes: int
    proporcao_classes: pd.Series


def carregar_dataset_bruto() -> pd.DataFrame:
    config.DIR_DADOS_RAW.mkdir(parents=True, exist_ok=True)
    dados = fetch_openml(
        config.OPENML_NOME_DATASET,
        version=config.OPENML_VERSAO_DATASET,
        as_frame=True,
        data_home=str(config.DIR_DADOS_RAW),
    )
    return dados.frame.copy()


def checar_dataset(df: pd.DataFrame) -> ChecagemDataset:
    colunas_categoricas = list(
        df.drop(columns=[config.COLUNA_ALVO]).select_dtypes(include=["category", "object"]).columns
    )
    colunas_numericas = list(
        df.drop(columns=[config.COLUNA_ALVO]).select_dtypes(include=["int64", "float64"]).columns
    )
    return ChecagemDataset(
        n_linhas=len(df),
        n_colunas=df.shape[1],
        colunas_categoricas=colunas_categoricas,
        colunas_numericas=colunas_numericas,
        n_valores_faltantes=int(df.isnull().sum().sum()),
        proporcao_classes=df[config.COLUNA_ALVO].value_counts(normalize=True),
    )


def preparar_para_c45(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia o alvo para "Decision" e converte colunas de texto para dtype
    `object` puro — a chefboost quebra com o dtype `string`/`category` do
    pandas moderno (ver docs/DECISOES.md, item 1)."""
    df_c45 = df.rename(columns={config.COLUNA_ALVO: "Decision"}).copy()
    colunas_texto = df_c45.select_dtypes(include=["category", "object"]).columns
    df_c45[colunas_texto] = df_c45[colunas_texto].astype(object)
    return df_c45


def preparar_para_cart(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot em vez de codificação ordinal: o CART só corta binário-numérico,
    e uma ordem ordinal entre categorias sem ordem real produziria regras sem sentido."""
    colunas_categoricas = df.drop(columns=[config.COLUNA_ALVO]).select_dtypes(
        include=["category", "object"]
    ).columns
    return pd.get_dummies(df, columns=list(colunas_categoricas), drop_first=False)


def dividir_treino_teste(df: pd.DataFrame, coluna_alvo: str, semente: int = config.SEMENTE_PRINCIPAL):
    return train_test_split(
        df,
        test_size=config.PROPORCAO_TESTE,
        stratify=df[coluna_alvo],
        random_state=semente,
    )
