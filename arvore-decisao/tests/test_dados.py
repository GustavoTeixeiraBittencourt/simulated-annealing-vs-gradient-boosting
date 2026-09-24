"""Testes de carga, checagem e divisão do dataset German Credit.

Requer acesso à internet na primeira execução (baixa o dataset do OpenML);
depois fica em cache local em data/raw/.
"""

import pandas as pd
import pytest

from src.arvore import config
from src.arvore.dados import (
    carregar_dataset_bruto,
    checar_dataset,
    dividir_treino_teste,
    preparar_para_c45,
    preparar_para_cart,
)


@pytest.fixture(scope="module")
def df_bruto() -> pd.DataFrame:
    return carregar_dataset_bruto()


def test_formato_esperado(df_bruto):
    checagem = checar_dataset(df_bruto)
    assert checagem.n_linhas == 1000
    assert checagem.n_colunas == 21  # 20 atributos + alvo
    assert checagem.n_valores_faltantes == 0
    assert len(checagem.colunas_categoricas) == 13
    assert len(checagem.colunas_numericas) == 7


def test_proporcao_de_classes_e_aproximadamente_70_30(df_bruto):
    checagem = checar_dataset(df_bruto)
    proporcao_boa = checagem.proporcao_classes["good"]
    assert 0.65 < proporcao_boa < 0.75


def test_divisao_e_estratificada(df_bruto):
    treino, teste = dividir_treino_teste(df_bruto, config.COLUNA_ALVO)
    proporcao_treino = treino[config.COLUNA_ALVO].value_counts(normalize=True)
    proporcao_teste = teste[config.COLUNA_ALVO].value_counts(normalize=True)
    for classe in proporcao_treino.index:
        assert proporcao_treino[classe] == pytest.approx(proporcao_teste[classe], abs=0.03)


def test_divisao_nao_tem_vazamento(df_bruto):
    treino, teste = dividir_treino_teste(df_bruto, config.COLUNA_ALVO)
    assert len(set(treino.index) & set(teste.index)) == 0
    assert len(treino) + len(teste) == len(df_bruto)


def test_preparar_para_c45_usa_dtype_object_em_colunas_de_texto(df_bruto):
    df_c45 = preparar_para_c45(df_bruto)
    assert "Decision" in df_c45.columns
    assert config.COLUNA_ALVO not in df_c45.columns
    colunas_texto = df_c45.drop(columns=["Decision"]).select_dtypes(include="object").columns
    for coluna in colunas_texto:
        assert df_c45[coluna].dtype == object
    # colunas numéricas continuam numéricas, para a chefboost tratá-las como contínuas
    for coluna in ["duration", "credit_amount", "age"]:
        assert pd.api.types.is_numeric_dtype(df_c45[coluna])


def test_preparar_para_cart_gera_apenas_colunas_numericas(df_bruto):
    df_cart = preparar_para_cart(df_bruto)
    colunas_atributos = df_cart.drop(columns=[config.COLUNA_ALVO])
    for coluna in colunas_atributos.columns:
        assert pd.api.types.is_numeric_dtype(colunas_atributos[coluna]) or pd.api.types.is_bool_dtype(
            colunas_atributos[coluna]
        )
