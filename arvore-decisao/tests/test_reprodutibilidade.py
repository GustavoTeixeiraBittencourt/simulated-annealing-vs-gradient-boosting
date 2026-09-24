"""Mesma semente, mesmos resultados — condição básica para que a arguição
possa reproduzir qualquer número citado no notebook."""

from sklearn.tree import DecisionTreeClassifier

from src.arvore import config
from src.arvore.dados import carregar_dataset_bruto, dividir_treino_teste, preparar_para_cart
from src.arvore.c45 import treinar_c45


def test_divisao_treino_teste_e_reprodutivel():
    df = carregar_dataset_bruto()
    treino_1, teste_1 = dividir_treino_teste(df, config.COLUNA_ALVO, semente=config.SEMENTE_PRINCIPAL)
    treino_2, teste_2 = dividir_treino_teste(df, config.COLUNA_ALVO, semente=config.SEMENTE_PRINCIPAL)
    assert list(treino_1.index) == list(treino_2.index)
    assert list(teste_1.index) == list(teste_2.index)


def test_cart_e_reprodutivel_com_mesma_semente():
    df = carregar_dataset_bruto()
    df_cart = preparar_para_cart(df)
    x = df_cart.drop(columns=[config.COLUNA_ALVO])
    y = df_cart[config.COLUNA_ALVO]

    arvore_1 = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=config.SEMENTE_PRINCIPAL)
    arvore_2 = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=config.SEMENTE_PRINCIPAL)
    arvore_1.fit(x, y)
    arvore_2.fit(x, y)

    assert (arvore_1.predict(x) == arvore_2.predict(x)).all()
    assert arvore_1.get_n_leaves() == arvore_2.get_n_leaves()


def test_c45_e_deterministico_entre_execucoes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = carregar_dataset_bruto()
    from src.arvore.dados import preparar_para_c45

    df_c45 = preparar_para_c45(df)
    arvore_1 = treinar_c45(df_c45, profundidade_maxima=3)
    previsoes_1 = arvore_1.prever(df_c45)

    arvore_2 = treinar_c45(df_c45, profundidade_maxima=3)
    previsoes_2 = arvore_2.prever(df_c45)

    assert previsoes_1 == previsoes_2
