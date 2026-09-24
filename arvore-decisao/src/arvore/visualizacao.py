"""Gráficos e desenho das árvores usados no notebook."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree

from . import config


def salvar_figura(fig, nome_arquivo: str) -> None:
    config.DIR_RESULTADOS_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.DIR_RESULTADOS_FIGURAS / nome_arquivo, bbox_inches="tight", dpi=150)


def desenhar_arvore_cart(arvore, nomes_colunas: list, nomes_classes: list, titulo: str = ""):
    fig, eixo = plt.subplots(figsize=(20, 10))
    plot_tree(
        arvore,
        feature_names=nomes_colunas,
        class_names=nomes_classes,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=eixo,
    )
    eixo.set_title(titulo)
    return fig


def grafico_complexidade_vs_acuracia(candidatos: list, meta: float = config.META_ACURACIA):
    df = pd.DataFrame(
        [
            {
                "n_folhas": c.n_folhas,
                "profundidade": c.profundidade_real,
                "acuracia_cv": c.acuracia_cv,
                "atinge_meta": c.atinge_meta,
            }
            for c in candidatos
        ]
    )
    fig, eixo = plt.subplots(figsize=(9, 6))
    cores = df["atinge_meta"].map({True: "#2a9d8f", False: "#e76f51"})
    eixo.scatter(df["n_folhas"], df["acuracia_cv"], c=cores, s=60, edgecolor="black", linewidth=0.5)
    eixo.axhline(meta, color="gray", linestyle="--", label=f"Meta ({meta:.0%})")
    eixo.set_xlabel("Número de folhas (complexidade)")
    eixo.set_ylabel("Acurácia média (validação cruzada, treino)")
    eixo.set_title("Complexidade × acurácia — busca da árvore mais enxuta")
    eixo.legend()
    return fig


def tabela_candidatos_cart(candidatos: list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "profundidade_maxima": c.profundidade_maxima,
                "min_amostras_folha": c.min_amostras_folha,
                "n_folhas": c.n_folhas,
                "profundidade_real": c.profundidade_real,
                "acuracia_cv": round(c.acuracia_cv, 4),
                "atinge_meta": c.atinge_meta,
            }
            for c in candidatos
        ]
    ).sort_values(["atinge_meta", "n_folhas"], ascending=[False, True])
