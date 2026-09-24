"""Testes do wrapper de C4.5 (chefboost), usando o Play Tennis como oráculo:
sabemos, pelo cálculo em test_metricas_divisao.py, que a raiz correta é
Outlook — se o wrapper não reproduzir isso, a integração está quebrada.
"""

import os

import pandas as pd
import pytest

from src.arvore.c45 import avaliar_c45, treinar_c45


@pytest.fixture
def play_tennis_chefboost(tmp_path, monkeypatch) -> pd.DataFrame:
    # A chefboost escreve outputs/rules/rules.py relativo ao cwd: isolamos
    # cada teste num diretório temporário para não sujar o repositório nem
    # misturar regras entre testes.
    monkeypatch.chdir(tmp_path)

    dados = {
        "Outlook": [
            "Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast",
            "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain",
        ],
        "Temperature": [
            "Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool",
            "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild",
        ],
        "Humidity": [
            "High", "High", "High", "High", "Normal", "Normal", "Normal",
            "High", "Normal", "Normal", "Normal", "High", "Normal", "High",
        ],
        "Wind": [
            "Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
            "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong",
        ],
        "Decision": [
            "No", "No", "Yes", "Yes", "Yes", "No", "Yes",
            "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No",
        ],
    }
    return pd.DataFrame(dados).astype(object)


def test_raiz_e_outlook(play_tennis_chefboost):
    arvore = treinar_c45(play_tennis_chefboost)
    texto = arvore.texto_das_regras()
    primeira_linha_com_feature = next(
        linha for linha in texto.splitlines() if '"feature"' in linha
    )
    assert '"feature": "Outlook"' in primeira_linha_com_feature
    assert '"depth": 1' in primeira_linha_com_feature


def test_profundidade_maxima_1_gera_unica_divisao(play_tennis_chefboost):
    arvore = treinar_c45(play_tennis_chefboost, profundidade_maxima=1)
    texto = arvore.texto_das_regras()
    linhas_com_feature = [linha for linha in texto.splitlines() if '"feature"' in linha]
    # com profundidade máxima 1, só pode existir a divisão da raiz
    assert len(linhas_com_feature) == 1


def test_previsoes_sao_deterministicas(play_tennis_chefboost):
    arvore = treinar_c45(play_tennis_chefboost)
    previsoes_1 = arvore.prever(play_tennis_chefboost)
    previsoes_2 = arvore.prever(play_tennis_chefboost)
    assert previsoes_1 == previsoes_2


def test_regras_cobrem_todos_os_caminhos(play_tennis_chefboost):
    arvore = treinar_c45(play_tennis_chefboost)
    acuracia = avaliar_c45(arvore, play_tennis_chefboost)
    # se alguma combinação de atributos não fosse coberta pelas regras, a
    # chefboost devolveria None e a comparação com o rótulo real falharia
    assert acuracia == pytest.approx(1.0)
