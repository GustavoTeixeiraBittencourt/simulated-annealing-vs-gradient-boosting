# Simulated Annealing vs. Gradient Boosting

### Um estudo comparativo de estratégias de buscas em inteligência artificial

**Professora:** Polyana Santos Fonseca Nascimento
**Autores:** Gustavo Teixeira Bittencourt de Oliveira, Edgar Klewert, Christophe Abelem, Adler Castro

---

## Sobre o projeto

Este repositório reúne a aplicação prática e a comparação crítica de duas abordagens de Inteligência Artificial vistas ao longo do bimestre: **Busca Local** e **Árvore de Decisão**. Em vez de apenas rodar os algoritmos, o objetivo foi entender por que cada um se comporta como se comporta, justificar as escolhas técnicas com evidência (não com "achismo"), e discutir honestamente os limites de cada resultado.

- Aplicar pelo menos duas modalidades de IA estudadas em disciplina a problemas realistas
- Justificar tecnicamente cada algoritmo e cada decisão de projeto
- Documentar parâmetros, decisões e limitações — inclusive quando o resultado não é o esperado
- Comparar as abordagens sob critérios equivalentes

---

## 1. Busca Local — Simulated Annealing

**Problema:** otimização de rota de entregas — o Problema do Caixeiro-Viajante (*Traveling Salesman Problem*), inspirado na necessidade real de uma transportadora minimizar a distância total percorrida ao visitar um conjunto de pontos de entrega.

**Algoritmo:** Simulated Annealing (Têmpera Simulada), com resfriamento geométrico.

**Por que essa escolha:** o espaço de busca do TSP é repleto de ótimos locais. Diferente do Hill Climbing puro — que trava assim que não encontra um vizinho melhor —, o Simulated Annealing aceita, com probabilidade decrescente ao longo do tempo, movimentos que pioram temporariamente a solução, permitindo escapar dessas armadilhas.

*(Implementação e resultados deste módulo em desenvolvimento pelo restante da equipe.)*

---

## 2. Árvore de Decisão — C4.5 vs. Gradient Boosting

**Problema:** concessão de crédito — classificar clientes como bom ou mau pagador a partir de seu histórico e situação financeira. Um problema real do setor financeiro, em que os dois tipos de erro têm custo bem diferente: aprovar crédito a um mau pagador custa mais caro do que recusar crédito a um bom pagador.

### Dataset

**[German Credit / Statlog](https://www.openml.org/search?type=data&id=31)** (`credit-g`, OpenML), com atributos legíveis — saldo em conta, histórico de crédito, duração e valor do empréstimo — e desbalanceamento moderado (70% bons / 30% maus pagadores).

O dataset originalmente cogitado, Credit Card Fraud (Kaggle), foi descartado: seus atributos são anonimizados por PCA (`V1`–`V28`, sem significado interpretável) e apenas ~0,17% das transações são fraude — desbalanceamento tão extremo que uma árvore sem nenhuma divisão já teria mais de 99% de acurácia, tornando qualquer meta de acurácia trivial e qualquer regra impossível de confrontar com conhecimento de domínio.

### Algoritmos comparados

| Algoritmo | Tipo | Biblioteca | Papel na comparação |
|---|---|---|---|
| **C4.5** | Árvore única, com razão de ganho | `chefboost` | Estratégia pedida pelo enunciado — árvore de decisão "clássica" |
| **Gradient Boosting** | Ensemble de árvores rasas treinadas sequencialmente | `scikit-learn` | Estratégia de ensemble, corrige o erro residual da anterior a cada passo |
| **CART (entropia)** | Árvore única | `scikit-learn` | Referência adicional — **não é C4.5** (cortes sempre binários, sem razão de ganho, poda diferente) |

### Metodologia

- Divisão treino/teste estratificada 80/20, com semente fixa para reprodutibilidade.
- Escolha de hiperparâmetros (profundidade, folhas mínimas) por validação cruzada **apenas no treino** — o conjunto de teste só é usado uma vez, no final, para não inflar o resultado.
- Meta de acurácia definida em **71%**, acima das duas baselines medidas (classe majoritária e árvore de um único nível, ambas em 70%) e dentro do teto empírico observado (~72%).
- Entre as configurações que atingem a meta, escolhe-se a **árvore mais enxuta** (menor número de folhas, depois menor profundidade) — o objetivo não é a árvore mais acurada possível, e sim a mais simples que já é "boa o suficiente".

### Resultados

| Modelo | Acurácia (teste) | Profundidade | Folhas / árvores | Regras legíveis? |
|---|---:|---:|---:|---|
| CART enxuto (referência) | 70,5% | 3 | 8 | Sim — poucas regras curtas |
| **C4.5** (chefboost) | 72,0% | 1 | 3 | Sim, mas não atinge a meta na validação |
| **Gradient Boosting** | 76,0% | 3 | 100 árvores | Não — só a importância agregada das variáveis é legível |

### O que esses números mostram

O Gradient Boosting vence em acurácia, como esperado de um ensemble — mas ao custo total de interpretabilidade: 100 árvores combinadas não podem ser lidas como um conjunto de regras, só resumidas por importância de variável.

O achado mais interessante, porém, veio do C4.5: apesar de fechar em 72% no teste, a busca por validação cruzada mostrou que sua acurácia **piora conforme a árvore cresce** (70,3% em profundidade 1, caindo até 67,9% em profundidade 5) — e nunca atinge de forma confiável a meta de 71%. A causa identificada: a implementação do C4.5 usada não faz poda real (só limita profundidade), e cada divisão categórica abre um ramo por categoria — com atributos de muitas categorias, isso fragmenta rapidamente o treino em subgrupos pequenos e ruidosos. O CART, com controle de mínimo de amostras por folha e cortes sempre binários, evita esse problema. Essa limitação não foi escondida: é reportada como parte legítima da comparação entre "C4.5 pronto de biblioteca" e "CART com controle fino de hiperparâmetros".

Também vale registrar que acurácia sozinha engana neste problema: a árvore CART mais enxuta, apesar de 70,5% de acurácia geral, acertou apenas 4 de 60 maus pagadores reais no teste — o modelo tende a sempre prever "bom pagador". Como aprovar crédito a quem não vai pagar custa mais caro que o contrário, essa limitação é discutida explicitamente em vez de ser mascarada pela acurácia agregada.

Todas as decisões técnicas deste módulo — com as verificações empíricas que as sustentam — estão em [`arvore-decisao/docs/DECISOES.md`](arvore-decisao/docs/DECISOES.md).

---

## Estrutura do repositório

```
.
├── busca-local/                    # Simulated Annealing (TSP) — em desenvolvimento
├── arvore-decisao/
│   ├── data/                       # cache do dataset German Credit (OpenML)
│   ├── src/arvore/                 # métricas de divisão, C4.5, CART, ensemble, avaliação
│   ├── notebooks/                  # notebook principal — a entrega desta modalidade
│   ├── tests/                      # testes automatizados (pytest)
│   ├── docs/                       # DECISOES.md e GUIA_ARGUICAO.md
│   └── resultados/                 # figuras e tabelas geradas
└── docs/                           # relatório final e análise comparativa
```

## Como reproduzir

```bash
cd arvore-decisao
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest tests/                       # 23 testes automatizados
jupyter notebook notebooks/arvore_decisao.ipynb   # notebook principal (roda também no Google Colab)
```

## Tecnologias

- Python 3.x
- scikit-learn (CART, Gradient Boosting)
- chefboost (C4.5)
- NumPy / Pandas
- Matplotlib (visualização de resultados)
- pytest (testes automatizados)

