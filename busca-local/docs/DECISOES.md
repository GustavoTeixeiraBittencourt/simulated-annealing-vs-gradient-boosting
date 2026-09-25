# Decisões de projeto — módulo Busca Local (Simulated Annealing / EVRP)

## Problema escolhido

O problema de Busca Local escolhido para este módulo é o **EVRP (Electric
Vehicle Routing Problem)** — otimização de rotas de entrega com veículos
elétricos, em que, além da distância percorrida, é preciso respeitar a
autonomia limitada da bateria e a possibilidade de recarga em pontos
específicos do trajeto. O algoritmo de otimização utilizado é o
**Simulated Annealing**.

## Datasets EVRP avaliados e descartados

Antes de adotar o GR17, pesquisamos datasets EVRP prontos, mas nenhum se
encaixava no escopo e prazo do trabalho:

- **Schneider, Stenger & Goeke (2014)**, *"The Electric Vehicle-Routing
  Problem with Time Windows and Recharging Stations"*, Transportation
  Science 48(4). O dataset E-VRPTW derivado desse trabalho inclui janelas de
  tempo por cliente, o que adiciona uma dimensão de complexidade (restrições
  temporais combinadas com restrições de bateria) além do que o módulo
  precisa demonstrar.
- **EC-TSP dataset (Mendeley)**, de Gialos & Zeimpekis (2023). Distribuído em
  formato Excel com múltiplas dimensões de dados (coordenadas, consumo,
  tempos, etc.), o que exigiria um trabalho de parsing e limpeza
  desproporcional ao tempo disponível para o módulo.

## Dataset adotado: GR17 (TSPLIB) adaptado para EVRP

Com autorização explícita da professora para adaptar um dataset existente
(desde que documentado), adotamos o **GR17** (TSPLIB, Groetschel), um TSP
clássico de 17 cidades com matriz de distância real e tour ótimo conhecido
(**2085**). O GR17 foi adaptado para EVRP por meio de um **mapeamento manual
de papéis** aos nós já existentes — nenhuma cidade foi adicionada, removida
ou reposicionada.

### Tabela de mapeamento

| Papel               | Nós (1-indexado) |
|---------------------|------------------|
| Depósito            | 1                |
| Estações de recarga | 3, 9, 14         |
| Clientes            | 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17 |

**Justificativa:** essas escolhas distribuem as estações de recarga ao longo
do tour ótimo de referência (1 → 4 → 13 → 7 → 8 → 6 → 17 → 14 → 15 → 3 → 11 →
10 → 2 → 5 → 9 → 12 → 16), evitando concentração das estações em um só
trecho do percurso.

### Parâmetros do modelo de bateria

| Parâmetro           | Valor  | Justificativa |
|---------------------|--------|----------------|
| `BATTERY_CAPACITY`  | 800    | A maior aresta do grafo é 745 (distância 2↔16); uma capacidade de 800 garante que qualquer trecho isolado entre dois nós seja sempre viável com bateria cheia. |
| `CONSUMPTION_RATE`  | 1.0    | Consumo de bateria 1:1 com a distância percorrida (mesma unidade da matriz de distância, sem conversão física). |
| `RECHARGE_MODE`     | "full" | A recarga é sempre total ao passar por uma estação ou pelo depósito, simplificando o modelo de bateria mantendo o problema não-trivial. |

A distância total do tour ótimo (~2085) é bem maior que a capacidade da
bateria (800), o que força pelo menos duas recargas ao longo do percurso —
garantindo que o aspecto de roteamento com restrição de autonomia seja de
fato relevante para o problema, e não apenas um TSP disfarçado.

### Nota sobre a dificuldade de viabilidade (achado empírico, e sua correção)

Um ponto importante, descoberto ao implementar e testar o Hill Climbing
(`src/busca_local/hill_climbing.py`): a recarga só ocorre ao **chegar** a um
depósito/estação — não ao passar por um cliente. Isso significa que um
cliente "encaixado" entre duas paradas de recarga consome a **soma** das
duas pernas (entrada + saída) de uma única carga, não cada perna
isoladamente. Como a distância média de aresta no GR17 é ~275 (com máximo de
745), essa soma frequentemente ultrapassa os 800 de capacidade — inclusive
no próprio tour ótimo original (testado e confirmado inviável no trecho
3 → 11 → 10 → 2 → 5 → 9, que acumula 1118 de distância sem recarga
intermediária).

Confirmamos por construção manual que **soluções viáveis existem** no espaço
de busca (ex.: uma rota "cata-vento", onde cada cliente é visitado em uma
ida-e-volta a partir da sua estação/depósito mais próximo, é viável para
todos os 13 clientes, embora com custo de distância bem pior que o ótimo).

**Causa raiz identificada e corrigida:** na primeira versão, `route_penalty`
retornava um valor em **degrau** — `INFEASIBLE_PENALTY` fixo (1e6) somado
apenas ao índice do trecho quebrado (0-16) — o que fazia toda rota inviável
custar ~1.000.000 independentemente de estar "quase viável" ou "muito
inviável". Sem gradiente entre essas duas situações, o Hill Climbing (que só
aceita vizinhos melhores-ou-iguais) nunca encontrava uma solução viável em
milhares de iterações (testado com até 200 reinícios × 5000 iterações = 1M+
avaliações, 0 execuções viáveis): eliminar a inviabilidade exige inserir
duas estações ao redor do mesmo cliente em sequência, e cada inserção
isolada aumentava o custo de distância sem que a penalidade de degrau desse
qualquer sinal de progresso — um "vale" de custo que a regra de aceitação
estrita do HC não conseguia atravessar.

A correção substituiu a penalidade de degrau por uma **proporcional ao
déficit total de bateria** (`evrp_mapping.compute_total_battery_deficit` +
`INFEASIBLE_PENALTY_BASE = 1000`, ver `problem.route_penalty`): agora a
penalidade cresce continuamente com o quanto a rota está "fora do orçamento
de bateria", dando ao HC um gradiente real para seguir. O resultado foi
imediato: nas mesmas condições de teste, o HC passou de 0/20 para **20/20
execuções encontrando solução viável**, com custos finais entre ~2450 e
~2650 (próximos da distância do tour ótimo, 2085, mais a sobrecarga das
recargas necessárias). Esse episódio é um ótimo ponto de discussão na
arguição: mostra concretamente por que a modelagem da função objetivo
importa tanto quanto o algoritmo de busca em si.

O piloto exploratório de Simulated Annealing (fora de `src/`, em
`scratch_sa_pilot.py`) valida esse mesmo comportamento sob o algoritmo que
será o principal da modalidade.

## Reprodução do ambiente

```bash
cd busca-local
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

As versões de `numpy` e `pytest` estão fixadas em `requirements.txt`
(`numpy==2.5.3`, `pytest==9.1.1`), testadas com Python 3.12.3. O `.venv/` é
local a cada máquina e não é versionado (ver `.gitignore`); o que garante a
reprodução é o `requirements.txt` committado, não a pasta `.venv/` em si.

## Nota de integridade dos dados

> Nenhuma distância entre nós foi alterada ou inventada — apenas o papel de
> cada nó (cliente/estação/depósito) e os parâmetros de bateria foram
> definidos pela equipe.
