# Simulated Annealing vs. Gradient Boosting: Um Estudo Comparativo de Estratégias de Inteligência Artificial

Projeto bimestral da disciplina de **Inteligência Computacional** — CESUPA (Centro Universitário do Estado do Pará), Engenharia de Computação, Turma EC8MA.

**Autores:** Gustavo Teixeira Bittencourt de Oliveira, Edgar Klewert, Christophe Abelem, Adler Castro

---

## Sobre o projeto

Este repositório documenta a aplicação prática e a comparação de duas abordagens distintas de Inteligência Artificial estudadas ao longo do bimestre:

1. **Busca Local** (otimização heurística)
2. **Árvore de Decisão** (aprendizado supervisionado via ensemble)

O objetivo não é apenas implementar os algoritmos, mas compreender profundamente seu funcionamento interno, justificar sua adequação aos problemas escolhidos e analisar criticamente os resultados obtidos.

## Objetivos

- Proporcionar experiência prática com pelo menos duas modalidades de IA estudadas em disciplina
- Definir problemas reais (ou inspirados em necessidades reais) adequados a cada modalidade
- Selecionar e justificar tecnicamente os algoritmos utilizados
- Desenvolver, executar e documentar as soluções, registrando parâmetros e decisões
- Comparar os resultados obtidos sob critérios equivalentes

## Abordagens estudadas

### 1. Busca Local — Simulated Annealing

**Problema:** Otimização de rota de entregas — o Problema do Caixeiro-Viajante (*Traveling Salesman Problem — TSP*), inspirado na necessidade real de uma transportadora minimizar a distância total percorrida ao visitar um conjunto de pontos de entrega.

**Algoritmo:** Simulated Annealing (Têmpera Simulada), com resfriamento geométrico.

**Por que essa escolha:** o espaço de busca do TSP é repleto de ótimos locais. Diferente do Hill Climbing puro — que trava assim que não encontra um vizinho melhor —, o Simulated Annealing aceita, com probabilidade decrescente ao longo do tempo, movimentos que pioram temporariamente a solução, permitindo escapar de armadilhas locais.


### 2. Árvore de Decisão — Gradient Boosting (Ensemble)

**Problema:** Detecção de fraude em transações de cartão de crédito — problema real e crítico no setor financeiro, com forte desbalanceamento de classes.

**Algoritmo:** Gradient Boosting, um método de ensemble que combina múltiplas árvores de decisão treinadas sequencialmente, cada uma corrigindo o erro residual da anterior.

**Por que essa escolha:** o dataset utilizado é extremamente desbalanceado (~0,17% de transações fraudulentas). O mecanismo corretivo do boosting — que foca nos exemplos onde o modelo atual mais erra — tende a favorecer a identificação da classe minoritária (fraude) de forma mais eficaz que uma árvore isolada.

**Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle) — transações europeias anonimizadas.


## Estrutura do repositório

```
.
├── busca-local/
│   ├── data/              # instâncias do problema (coordenadas dos pontos)
│   ├── src/                # implementação do Simulated Annealing
│   └── results/             # gráficos de convergência, parâmetros testados, evidências
├── arvore-decisao/
│   ├── data/               # dataset de fraude
│   ├── src/                 # implementação/treinamento do Gradient Boosting
│   └── results/              # métricas, matriz de confusão, evidências
├── docs/                      # relatório final e análise comparativa
└── README.md
```

## Tecnologias

- Python 3.x
- scikit-learn (Gradient Boosting)
- NumPy / Pandas
- Matplotlib (visualização de resultados)


## Licença

Projeto acadêmico desenvolvido para fins educacionais na disciplina de Inteligência Computacional.