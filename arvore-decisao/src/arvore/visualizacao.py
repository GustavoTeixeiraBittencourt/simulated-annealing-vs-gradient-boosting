"""Gráficos e desenho das árvores usados no notebook."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree._export import _MPLTreeExporter
from sklearn.utils.validation import check_is_fitted

from . import config
from .regras import eh_feature_one_hot, nomes_perguntas_cart


class _ExportadorArvoreComSetasRotuladas(_MPLTreeExporter):
    """Igual ao exportador do `plot_tree` do scikit-learn, mas escreve
    "Sim"/"Não" em cima de TODAS as setas — o `plot_tree` original só rotula
    as duas setas que saem da raiz, deixando quem não é técnico sem saber
    para que lado ir nos nós internos."""

    def __init__(self, *args, nomes_colunas_originais, **kwargs):
        super().__init__(*args, **kwargs)
        self._nomes_colunas_originais = nomes_colunas_originais

    def _rotulo_da_seta(self, tree, node_pai, vai_para_esquerda: bool) -> str:
        nome_feature = self._nomes_colunas_originais[tree.feature[node_pai.tree.node_id]]
        # Feature one-hot: o nó pergunta "é a categoria X?", mas o corte é no
        # indicador 0/1 — "<= 0.5" é o indicador 0, ou seja, resposta "não".
        # Feature numérica: o nó já É a pergunta ("valor <= limiar?"), então
        # "<= limiar" (esquerda) é diretamente a resposta "sim".
        condicao_satisfeita_e_sim = not eh_feature_one_hot(nome_feature)
        vai_para_esquerda_e_sim = vai_para_esquerda == condicao_satisfeita_e_sim
        return "Sim" if vai_para_esquerda_e_sim else "Não"

    def recurse(self, node, tree, ax, max_x, max_y, depth=0):
        common_kwargs = dict(zorder=100 - 10 * depth, xycoords="axes fraction")
        if self.fontsize is not None:
            common_kwargs["fontsize"] = self.fontsize

        kwargs = dict(
            ha="center",
            va="center",
            bbox=self.bbox_args.copy(),
            arrowprops=self.arrow_args.copy(),
            **common_kwargs,
        )
        kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]

        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)

        if self.max_depth is not None and depth > self.max_depth:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            kwargs["bbox"]["fc"] = "grey"
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)
            return

        kwargs["bbox"]["fc"] = (
            self.get_fill_color(tree, node.tree.node_id) if self.filled else ax.get_facecolor()
        )

        if node.parent is None:
            ax.annotate(node.tree.label, xy, **kwargs)
        else:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            ax.annotate(node.tree.label, xy_parent, xy, **kwargs)

            vai_para_esquerda = node.parent.left() == node
            texto = self._rotulo_da_seta(tree, node.parent, vai_para_esquerda)
            text_pos = ((xy_parent[0] + xy[0]) / 2, (xy_parent[1] + xy[1]) / 2)
            label_ha = "right" if vai_para_esquerda else "left"
            texto = (texto + "  ") if vai_para_esquerda else ("  " + texto)
            ax.annotate(texto, text_pos, ha=label_ha, **common_kwargs)

        for child in node.children:
            self.recurse(child, tree, ax, max_x, max_y, depth=depth + 1)


def salvar_figura(fig, nome_arquivo: str) -> None:
    config.DIR_RESULTADOS_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.DIR_RESULTADOS_FIGURAS / nome_arquivo, bbox_inches="tight", dpi=150)


def desenhar_arvore_cart(arvore, nomes_colunas: list, nomes_classes: list, titulo: str = ""):
    """Desenha a árvore CART com as perguntas em português nos nós e a
    resposta ("Sim"/"Não") escrita em cima de cada seta, para que alguém sem
    conhecimento técnico consiga acompanhar cada divisão só olhando a
    imagem."""
    check_is_fitted(arvore)
    fig, eixo = plt.subplots(figsize=(22, 11))
    exportador = _ExportadorArvoreComSetasRotuladas(
        nomes_colunas_originais=nomes_colunas,
        feature_names=nomes_perguntas_cart(nomes_colunas),
        class_names=nomes_classes,
        filled=True,
        rounded=True,
        fontsize=8,
    )
    exportador.export(arvore, ax=eixo)
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
