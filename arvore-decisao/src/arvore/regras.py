"""Extração de regras das árvores (CART e C4.5) e tradução para português
claro, no formato "SE ... ENTÃO ...", incluindo os nomes técnicos de colunas
e categorias do German Credit."""

import re

import numpy as np
from sklearn.tree import _tree

TRADUCAO_ATRIBUTOS = {
    "checking_status": "saldo da conta corrente",
    "duration": "duração do empréstimo (meses)",
    "credit_history": "histórico de crédito",
    "purpose": "finalidade do empréstimo",
    "credit_amount": "valor do empréstimo",
    "savings_status": "saldo em poupança",
    "employment": "tempo no emprego atual",
    "installment_commitment": "parcela (% da renda)",
    "personal_status": "estado civil/sexo",
    "other_parties": "outros devedores/fiadores",
    "residence_since": "tempo na residência atual (anos)",
    "property_magnitude": "principal patrimônio",
    "age": "idade",
    "other_payment_plans": "outros planos de pagamento",
    "housing": "situação de moradia",
    "existing_credits": "nº de créditos existentes no banco",
    "job": "qualificação profissional",
    "num_dependents": "nº de dependentes",
    "own_telephone": "possui telefone",
    "foreign_worker": "trabalhador estrangeiro",
}

TRADUCAO_CATEGORIAS = {
    "checking_status": {
        "<0": "saldo negativo",
        "0<=X<200": "saldo entre 0 e 200 marcos",
        ">=200": "saldo de 200 marcos ou mais",
        "no checking": "sem conta corrente",
    },
    "credit_history": {
        "no credits/all paid": "nunca tomou crédito / já quitou tudo",
        "all paid": "quitou todos os créditos neste banco",
        "existing paid": "créditos existentes em dia",
        "delayed previously": "já atrasou pagamentos antes",
        "critical/other existing credit": "histórico crítico / créditos em outros bancos",
    },
    "savings_status": {
        "<100": "poupança abaixo de 100 marcos",
        "100<=X<500": "poupança entre 100 e 500 marcos",
        "500<=X<1000": "poupança entre 500 e 1000 marcos",
        ">=1000": "poupança de 1000 marcos ou mais",
        "no known savings": "sem poupança conhecida",
    },
    "employment": {
        "unemployed": "desempregado",
        "<1": "empregado há menos de 1 ano",
        "1<=X<4": "empregado entre 1 e 4 anos",
        "4<=X<7": "empregado entre 4 e 7 anos",
        ">=7": "empregado há 7 anos ou mais",
    },
    "housing": {"own": "casa própria", "rent": "aluguel", "for free": "moradia cedida"},
    "own_telephone": {"yes": "sim", "none": "não"},
    "foreign_worker": {"yes": "sim", "no": "não"},
    "property_magnitude": {
        "real estate": "imóvel",
        "life insurance": "seguro de vida",
        "car": "carro",
        "no known property": "sem patrimônio conhecido",
    },
    "purpose": {
        "new car": "carro novo",
        "used car": "carro usado",
        "furniture/equipment": "móveis/equipamentos",
        "radio/tv": "rádio/TV",
        "domestic appliance": "eletrodoméstico",
        "repairs": "reparos",
        "education": "educação",
        "retraining": "requalificação profissional",
        "business": "negócio próprio",
        "other": "outro",
    },
    "personal_status": {
        "male single": "homem solteiro",
        "female div/dep/mar": "mulher divorciada/dependente/casada",
        "male div/sep": "homem divorciado/separado",
        "male mar/wid": "homem casado/viúvo",
    },
    "other_parties": {
        "none": "nenhum",
        "guarantor": "fiador",
        "co applicant": "co-solicitante",
    },
    "other_payment_plans": {"none": "nenhum", "bank": "banco", "stores": "lojas"},
    "job": {
        "skilled": "qualificado",
        "unskilled resident": "não qualificado, residente",
        "unemp/unskilled non res": "desempregado/não qualificado, não residente",
        "high qualif/self emp/mgmt": "altamente qualificado/autônomo/gerência",
    },
}

TRADUCAO_CLASSES = {"good": "bom pagador", "bad": "mau pagador"}


def traduzir_atributo(nome_coluna: str) -> str:
    return TRADUCAO_ATRIBUTOS.get(nome_coluna, nome_coluna)


def traduzir_categoria(nome_coluna: str, valor) -> str:
    return TRADUCAO_CATEGORIAS.get(nome_coluna, {}).get(str(valor), str(valor))


def traduzir_classe(rotulo: str) -> str:
    return TRADUCAO_CLASSES.get(rotulo, rotulo)


def coluna_e_categoria_one_hot(nome_feature: str):
    """Se `nome_feature` veio do one-hot de `preparar_para_cart`, devolve
    `(coluna_original, categoria)`; caso contrário devolve `None`. Único
    lugar que sabe reconstruir o atributo/categoria a partir do nome da
    coluna gerada por `pandas.get_dummies` — reusado por quem descreve
    condições, monta perguntas e decide o sentido das setas na árvore."""
    for coluna_original in TRADUCAO_ATRIBUTOS:
        prefixo = coluna_original + "_"
        if nome_feature.startswith(prefixo):
            return coluna_original, nome_feature[len(prefixo):]
    return None


def eh_feature_one_hot(nome_feature: str) -> bool:
    return coluna_e_categoria_one_hot(nome_feature) is not None


def _descrever_condicao_cart(nome_feature: str, limiar: float, vai_para_esquerda: bool) -> str:
    """Quando a feature vem de one-hot, o corte em 0.5 vira uma afirmação
    categórica em vez de uma desigualdade numérica sem sentido."""
    par = coluna_e_categoria_one_hot(nome_feature)
    if par is not None:
        coluna_original, categoria = par
        descricao = f"{traduzir_atributo(coluna_original)} = {traduzir_categoria(coluna_original, categoria)}"
        return ("NÃO é o caso que " + descricao) if vai_para_esquerda else descricao

    nome_traduzido = traduzir_atributo(nome_feature)
    operador = "<=" if vai_para_esquerda else ">"
    return f"{nome_traduzido} {operador} {limiar:.1f}"


def extrair_regras_cart(arvore, nomes_colunas: list, nomes_classes: list) -> list:
    """Uma regra por folha, no formato "SE ... ENTÃO classe (n amostras)"."""
    arvore_interna = arvore.tree_
    regras = []

    def percorrer(no, condicoes):
        if arvore_interna.feature[no] != _tree.TREE_UNDEFINED:
            nome_feature = nomes_colunas[arvore_interna.feature[no]]
            limiar = arvore_interna.threshold[no]

            percorrer(
                arvore_interna.children_left[no],
                condicoes + [_descrever_condicao_cart(nome_feature, limiar, vai_para_esquerda=True)],
            )
            percorrer(
                arvore_interna.children_right[no],
                condicoes + [_descrever_condicao_cart(nome_feature, limiar, vai_para_esquerda=False)],
            )
        else:
            contagens = arvore_interna.value[no][0]
            classe_prevista = nomes_classes[int(np.argmax(contagens))]
            n_amostras = int(arvore_interna.n_node_samples[no])
            pureza = contagens.max() / contagens.sum()
            regras.append(
                {
                    "condicoes": condicoes,
                    "classe": traduzir_classe(classe_prevista),
                    "n_amostras": n_amostras,
                    "pureza": pureza,
                }
            )

    percorrer(0, [])
    return regras


def nomes_perguntas_cart(nomes_colunas: list) -> list:
    """Traduz os nomes técnicos das colunas one-hot/numéricas do CART em
    rótulos legíveis (perguntas de sim/não, ou o nome do atributo em
    português) para usar como `feature_names` do `plot_tree` — sem isso, a
    árvore desenhada mostra nomes de coluna crus como "checking_status_no
    checking", ilegíveis para quem não é técnico."""
    perguntas = []
    for nome in nomes_colunas:
        par = coluna_e_categoria_one_hot(nome)
        if par is not None:
            coluna_original, categoria = par
            descricao = traduzir_categoria(coluna_original, categoria)
            pergunta = f"{traduzir_atributo(coluna_original).capitalize()} é \"{descricao}\"?"
        else:
            pergunta = traduzir_atributo(nome).capitalize()
        perguntas.append(pergunta)
    return perguntas


def caminho_decisao_cart(arvore, linha, nomes_colunas: list) -> list:
    """Caminho, nó a nó, que uma linha percorre na árvore CART até a folha."""
    arvore_interna = arvore.tree_
    caminho = []
    no = 0
    while arvore_interna.feature[no] != _tree.TREE_UNDEFINED:
        indice_feature = arvore_interna.feature[no]
        nome_feature = nomes_colunas[indice_feature]
        limiar = arvore_interna.threshold[no]
        valor = linha[nome_feature]
        vai_para_esquerda = valor <= limiar
        caminho.append(_descrever_condicao_cart(nome_feature, limiar, vai_para_esquerda))
        no = arvore_interna.children_left[no] if vai_para_esquerda else arvore_interna.children_right[no]
    return caminho


def formatar_regra(regra: dict) -> str:
    condicoes = " E ".join(regra["condicoes"])
    return (
        f"SE {condicoes} "
        f"ENTÃO {regra['classe']} "
        f"({regra['n_amostras']} amostras no treino, {regra['pureza']:.0%} de pureza)"
    )


_PADRAO_CONDICAO_C45 = re.compile(
    r"obj\[(?P<indice>\d+)\](?P<operador>==|<=|>)\s*'?(?P<valor>[^':]+?)'?:"
)


def extrair_regras_c45(texto_regras: str, colunas_atributos: list) -> list:
    """Parseia o `rules.py` gerado pela chefboost (if/elif aninhados) para o
    mesmo formato de `extrair_regras_cart`."""
    regras = []
    pilha_condicoes = []

    for linha in texto_regras.splitlines():
        linha_sem_comentario = linha.split("#")[0]
        if not linha_sem_comentario.strip():
            continue
        indentacao = len(linha) - len(linha.lstrip(" "))

        pilha_condicoes = [c for c in pilha_condicoes if c[0] < indentacao]

        casamento = _PADRAO_CONDICAO_C45.search(linha_sem_comentario)
        if casamento:
            indice = int(casamento.group("indice"))
            operador = casamento.group("operador")
            valor = casamento.group("valor").strip()
            coluna = colunas_atributos[indice]

            if operador == "==":
                descricao = f"{traduzir_atributo(coluna)} = {traduzir_categoria(coluna, valor)}"
            else:
                try:
                    valor_formatado = f"{float(valor):.1f}"
                except ValueError:
                    valor_formatado = valor
                descricao = f"{traduzir_atributo(coluna)} {operador} {valor_formatado}"
            pilha_condicoes.append((indentacao, descricao))
            continue

        if "return" in linha_sem_comentario:
            classe = linha_sem_comentario.split("return")[1].strip().strip("'\"")
            regras.append(
                {
                    "condicoes": [c[1] for c in pilha_condicoes],
                    "classe": traduzir_classe(classe),
                    "n_amostras": None,
                    "pureza": None,
                }
            )

    return regras


def formatar_regra_c45(regra: dict) -> str:
    condicoes = " E ".join(regra["condicoes"]) if regra["condicoes"] else "(sempre)"
    return f"SE {condicoes} ENTÃO {regra['classe']}"
