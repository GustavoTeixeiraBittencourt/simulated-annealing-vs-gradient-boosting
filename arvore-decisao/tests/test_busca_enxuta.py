"""Confere que a árvore escolhida pela busca enxuta realmente atinge a meta
na validação cruzada, e que nenhuma árvore mais simples testada a atinge."""

import pytest

from src.arvore import config
from src.arvore.busca_enxuta import buscar_arvore_enxuta_cart
from src.arvore.dados import carregar_dataset_bruto, dividir_treino_teste, preparar_para_cart


@pytest.fixture(scope="module")
def dados_treino_cart():
    df = carregar_dataset_bruto()
    treino, _ = dividir_treino_teste(df, config.COLUNA_ALVO)
    df_cart = preparar_para_cart(df)
    treino_cart = df_cart.loc[treino.index]
    x_treino = treino_cart.drop(columns=[config.COLUNA_ALVO])
    y_treino = treino_cart[config.COLUNA_ALVO]
    return x_treino, y_treino


def test_melhor_candidato_atinge_a_meta(dados_treino_cart):
    x_treino, y_treino = dados_treino_cart
    melhor, _ = buscar_arvore_enxuta_cart(x_treino, y_treino)
    assert melhor.acuracia_cv >= config.META_ACURACIA


def test_nenhuma_arvore_mais_simples_atinge_a_meta(dados_treino_cart):
    x_treino, y_treino = dados_treino_cart
    melhor, todos = buscar_arvore_enxuta_cart(x_treino, y_treino)

    mais_simples_que_o_melhor = [
        c
        for c in todos
        if (c.n_folhas, c.profundidade_real) < (melhor.n_folhas, melhor.profundidade_real)
    ]
    for candidato in mais_simples_que_o_melhor:
        assert not candidato.atinge_meta
