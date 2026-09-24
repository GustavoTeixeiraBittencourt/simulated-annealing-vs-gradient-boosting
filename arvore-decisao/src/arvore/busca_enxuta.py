"""Busca da árvore mais enxuta que atinge `config.META_ACURACIA` (ver docs/DECISOES.md,
itens 2-4). Seleção por validação cruzada no treino; entre as que atingem a
meta, escolhe-se a menos complexa (menor nº de folhas, depois menor profundidade)."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from . import config
from .c45 import avaliar_c45, treinar_c45


@dataclass
class CandidatoArvore:
    profundidade_maxima: object
    min_amostras_folha: int
    acuracia_cv: float
    n_folhas: int
    profundidade_real: int
    atinge_meta: bool


def _chave_de_desempate(candidato: CandidatoArvore):
    return (candidato.n_folhas, candidato.profundidade_real)


def buscar_arvore_enxuta_cart(x_treino: pd.DataFrame, y_treino: pd.Series) -> tuple[CandidatoArvore, list]:
    dobras = StratifiedKFold(
        n_splits=config.DOBRAS_VALIDACAO_CRUZADA, shuffle=True, random_state=config.SEMENTE_PRINCIPAL
    )
    candidatos = []
    for profundidade in config.GRADE_PROFUNDIDADE_MAXIMA:
        for min_folha in config.GRADE_MIN_AMOSTRAS_FOLHA:
            arvore = DecisionTreeClassifier(
                criterion="entropy",
                max_depth=profundidade,
                min_samples_leaf=min_folha,
                random_state=config.SEMENTE_PRINCIPAL,
            )
            acuracias = cross_val_score(arvore, x_treino, y_treino, cv=dobras, scoring="accuracy")
            arvore.fit(x_treino, y_treino)
            candidatos.append(
                CandidatoArvore(
                    profundidade_maxima=profundidade,
                    min_amostras_folha=min_folha,
                    acuracia_cv=acuracias.mean(),
                    n_folhas=arvore.get_n_leaves(),
                    profundidade_real=arvore.get_depth(),
                    atinge_meta=acuracias.mean() >= config.META_ACURACIA,
                )
            )

    candidatos_na_meta = [c for c in candidatos if c.atinge_meta]
    if not candidatos_na_meta:
        raise ValueError(
            "Nenhuma configuração testada atingiu a meta de acurácia "
            f"({config.META_ACURACIA:.0%}) na validação cruzada. Ver docs/DECISOES.md."
        )
    melhor = min(candidatos_na_meta, key=_chave_de_desempate)
    return melhor, candidatos


@dataclass
class CandidatoC45:
    profundidade_maxima: object
    acuracia_cv: float
    n_folhas: int
    profundidade_real: int
    atinge_meta: bool


def _estrutura_da_arvore_c45(texto_regras: str) -> tuple[int, int]:
    n_folhas = sum(1 for linha in texto_regras.splitlines() if linha.strip().startswith("return"))
    profundidades = [
        int(linha.split('"depth":')[1].split("}")[0])
        for linha in texto_regras.splitlines()
        if '"depth":' in linha
    ]
    profundidade_real = max(profundidades) if profundidades else 0
    return n_folhas, profundidade_real


def buscar_arvore_enxuta_c45(df_treino_c45: pd.DataFrame) -> tuple[CandidatoC45, list]:
    """Mesma lógica de `buscar_arvore_enxuta_cart`, mas para o C4.5 (chefboost).

    A busca é 1-D (só profundidade) porque a chefboost não expõe
    `min_samples_leaf` nem poda pós-treino. Neste dataset a acurácia piora
    com a profundidade (sem poda real, ver docs/DECISOES.md) e nenhum
    candidato atinge a meta — por isso a função não lança exceção, apenas
    devolve o melhor candidato com `atinge_meta=False`.
    """
    dobras = StratifiedKFold(
        n_splits=config.DOBRAS_VALIDACAO_CRUZADA, shuffle=True, random_state=config.SEMENTE_PRINCIPAL
    )
    candidatos = []
    for profundidade in config.GRADE_PROFUNDIDADE_MAXIMA_C45:
        acuracias_das_dobras = []
        for indice_treino, indice_val in dobras.split(df_treino_c45, df_treino_c45["Decision"]):
            arvore_dobra = treinar_c45(df_treino_c45.iloc[indice_treino], profundidade_maxima=profundidade)
            acuracias_das_dobras.append(avaliar_c45(arvore_dobra, df_treino_c45.iloc[indice_val]))

        arvore_completa = treinar_c45(df_treino_c45, profundidade_maxima=profundidade)
        n_folhas, profundidade_real = _estrutura_da_arvore_c45(arvore_completa.texto_das_regras())
        acuracia_media = sum(acuracias_das_dobras) / len(acuracias_das_dobras)
        candidatos.append(
            CandidatoC45(
                profundidade_maxima=profundidade,
                acuracia_cv=acuracia_media,
                n_folhas=n_folhas,
                profundidade_real=profundidade_real,
                atinge_meta=acuracia_media >= config.META_ACURACIA,
            )
        )

    candidatos_na_meta = [c for c in candidatos if c.atinge_meta]
    if candidatos_na_meta:
        melhor = min(candidatos_na_meta, key=lambda c: (c.n_folhas, c.profundidade_real))
    else:
        melhor = max(candidatos, key=lambda c: c.acuracia_cv)
    return melhor, candidatos
