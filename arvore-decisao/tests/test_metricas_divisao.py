"""Valida entropia/ganho/split-info/razão-de-ganho contra o exemplo clássico
Play Tennis (Quinlan) e casos-limite simples.

Valores de referência (Play Tennis, atributo Outlook): entropia inicial
≈ 0,940; ganho ≈ 0,246; split info ≈ 1,577; razão de ganho ≈ 0,156.
"""

import math

import pandas as pd
import pytest

from src.arvore.metricas_divisao import (
    entropia,
    ganho_informacao,
    melhor_atributo_por_razao_de_ganho,
    razao_de_ganho,
    split_information,
)

TOLERANCIA = 1e-3


@pytest.fixture
def play_tennis() -> pd.DataFrame:
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
    return pd.DataFrame(dados)


def test_entropia_inicial_play_tennis(play_tennis):
    assert entropia(play_tennis["Decision"]) == pytest.approx(0.940, abs=TOLERANCIA)


def test_entropia_grupo_puro_e_zero():
    rotulos = pd.Series(["Yes", "Yes", "Yes"])
    assert entropia(rotulos) == pytest.approx(0.0, abs=TOLERANCIA)


def test_entropia_50_50_e_um():
    rotulos = pd.Series(["Yes", "No"])
    assert entropia(rotulos) == pytest.approx(1.0, abs=TOLERANCIA)


def test_ganho_informacao_outlook(play_tennis):
    ganho = ganho_informacao(play_tennis, "Outlook", "Decision")
    assert ganho == pytest.approx(0.246, abs=TOLERANCIA)


def test_split_information_outlook(play_tennis):
    si = split_information(play_tennis, "Outlook")
    assert si == pytest.approx(1.577, abs=TOLERANCIA)


def test_razao_de_ganho_outlook(play_tennis):
    razao = razao_de_ganho(play_tennis, "Outlook", "Decision")
    assert razao == pytest.approx(0.156, abs=TOLERANCIA)


def test_atributo_identificador_tem_ganho_maximo_mas_razao_penalizada(play_tennis):
    """Um atributo tipo ID (um valor distinto por linha) maximiza o ganho de
    informação (separa cada linha em seu próprio grupo puro), mas a razão de
    ganho penaliza isso — motivo de a razão de ganho existir no C4.5.

    Repetimos o Play Tennis 10 vezes (mantendo as mesmas proporções, logo o
    ganho e a razão de ganho de Outlook não mudam) só para dar ao atributo
    ID categorias suficientes (140) para que a penalização do split
    information supere seu ganho bruto máximo e inverta o ranking. Com as
    14 linhas originais isso ainda não aconteceria: a penalização já reduz
    bastante a razão de ganho do ID, mas não o suficiente para ultrapassar
    Outlook num dataset tão pequeno — o que por si só já mostra que a força
    da penalização cresce com o número de instâncias, como esperado.
    """
    df = pd.concat([play_tennis] * 10, ignore_index=True)
    df["ID"] = [f"cliente_{i}" for i in range(len(df))]

    ganho_id = ganho_informacao(df, "ID", "Decision")
    ganho_outlook = ganho_informacao(df, "Outlook", "Decision")
    assert ganho_id > ganho_outlook
    assert ganho_id == pytest.approx(entropia(df["Decision"]), abs=TOLERANCIA)

    razao_id = razao_de_ganho(df, "ID", "Decision")
    razao_outlook = razao_de_ganho(df, "Outlook", "Decision")
    assert razao_id < razao_outlook


def test_melhor_atributo_por_razao_de_ganho_escolhe_outlook(play_tennis):
    ranking = melhor_atributo_por_razao_de_ganho(
        play_tennis, ["Outlook", "Temperature", "Humidity", "Wind"], "Decision"
    )
    melhor = next(iter(ranking))
    assert melhor == "Outlook"
