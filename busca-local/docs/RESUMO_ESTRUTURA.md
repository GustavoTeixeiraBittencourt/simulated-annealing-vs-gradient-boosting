# Resumo da estrutura — módulo Busca Local (EVRP / GR17)

Documento de referência gerado a partir da leitura direta de todos os
arquivos citados (código-fonte, dados, CSVs de resultados) — nenhum
conteúdo foi inferido sem checar o arquivo correspondente.

## Seção 1 — Visão geral

Este módulo implementa e compara três algoritmos de Busca Local — **Hill
Climbing com reinício aleatório**, **Simulated Annealing** e **Simulated
Annealing com reaquecimento** (variante aprimorada) — aplicados a uma
instância de **EVRP** (Electric Vehicle Routing Problem, roteamento de
veículos elétricos com restrição de autonomia de bateria e recarga em
estações) construída a partir do dataset **GR17** (TSPLIB, 17 cidades),
adaptado por mapeamento manual de papéis (depósito, estações, clientes) aos
nós existentes — decisão justificada em `docs/DECISOES.md`. O pacote Python
com a implementação de produção vive em `src/busca_local/`, coberto por uma
suíte de 37 testes automatizados em `tests/`. A **entrega principal** do
módulo é o notebook `notebooks/simulated_annealing_evrp.ipynb`, que consolida
a definição do problema, a justificativa dos algoritmos, os experimentos e a
comparação estatística final entre os três algoritmos (Wilcoxon pareado,
n=30), usando como fonte de dados os CSVs em `resultados/`.

## Seção 2 — Estrutura de pastas

- **`data/raw/`** — contém os três arquivos originais do dataset GR17
  (`gr17.tsp`, `gr17_d.txt`, `gr17_s.txt`): a matriz de distância completa
  17×17, o tour ótimo de referência e o arquivo TSPLIB original. São os
  únicos arquivos de dados de entrada do módulo; nenhum valor neles foi
  editado manualmente (ver `docs/DECISOES.md`).
- **`data/processed/`** — **vazia** (contém apenas um `.gitkeep` para o
  Git rastrear a pasta). Isso é intencional, não um esquecimento: o
  pipeline atual (`dataset.load_distance_matrix` / `load_optimal_tour`)
  carrega e valida o dataset em memória a cada execução, diretamente de
  `data/raw/`, sem persistir uma versão pré-processada em disco. Não há
  necessidade disso neste projeto porque o dataset é pequeno (matriz
  17×17) e o carregamento é praticamente instantâneo — criar uma etapa de
  pré-processamento persistido só adicionaria complexidade sem ganho de
  desempenho perceptível. Se o dataset crescesse (ou se o
  carregamento/validação se tornasse custoso), seria aqui que uma versão
  processada e cacheada em disco entraria.
- **`src/busca_local/`** — o pacote Python principal da modalidade, com 9
  módulos `.py` (mais o `__init__.py`) implementando o problema, a
  vizinhança, os três algoritmos de busca e a infraestrutura de
  experimentação. Descrito módulo a módulo na Seção 3.
- **`notebooks/`** — contém o notebook de entrega principal,
  `simulated_annealing_evrp.ipynb` (mais um `.gitkeep`).
- **`tests/`** — a suíte de testes automatizados (`pytest`), 6 arquivos,
  37 testes no total. Detalhada na Seção 6.
- **`docs/`** — documentação de decisões do projeto: `DECISOES.md` (o
  histórico de decisões de modelagem e calibração) e este próprio
  `RESUMO_ESTRUTURA.md`.
- **`resultados/`** — 5 arquivos `.csv` gerados pelos scripts de
  experimento oficiais na raiz de `busca-local/` (mais um `.gitkeep`).
  Detalhados na Seção 5.

## Seção 3 — Cada módulo em `src/busca_local/`

| Arquivo | O que define | Para que serve no fluxo geral |
|---|---|---|
| `dataset.py` | `DatasetValidationError`; `OptimalTour` (dataclass); `load_distance_matrix()`; `load_optimal_tour()` | Carrega e valida a matriz de distância e o tour ótimo do GR17 a partir de `data/raw/`, com checagem de simetria/diagonal zero. Usado por todos os algoritmos e scripts como fonte de dados. |
| `evrp_mapping.py` | Constantes `DEPOT`, `CHARGING_STATIONS`, `CUSTOMERS`, `BATTERY_CAPACITY`, `CONSUMPTION_RATE`, `RECHARGE_MODE`; `is_feasible_route()`; `compute_total_battery_deficit()` | Define o mapeamento EVRP sobre o GR17 e a física de bateria: simula o consumo/recarga ao longo de uma rota. `is_feasible_route` responde sim/não (parando no primeiro trecho quebrado); `compute_total_battery_deficit` continua a simulação e soma o déficit de TODOS os trechos, servindo de sinal de gradiente para a penalidade em `problem.py`. |
| `problem.py` | `Route` (alias de tipo); `INFEASIBLE_PENALTY_BASE`; `STATION_INSERTION_PROBABILITY`; `route_cost()`; `route_penalty()`; `evaluate()`; `random_valid_route()` | Define a representação de uma rota e a **função objetivo única** (`evaluate = route_cost + route_penalty`) usada pelos três algoritmos, além do gerador de rotas iniciais aleatórias. |
| `neighborhood.py` | `swap_neighbor()`; `insert_station_neighbor()`; `remove_station_neighbor()`; `get_neighbor()` | Implementa as três operações de vizinhança (troca de nós, inserção/remoção de estação) e a função que escolhe uma delas aleatoriamente — reusada pelos três algoritmos de busca. |
| `hill_climbing.py` | `HillClimbingResult` (dataclass); `hill_climbing()`; `hill_climbing_fixed_budget()` | Implementa o Hill Climbing com reinício aleatório em duas variantes: `hill_climbing()` (parada por estagnação, algoritmo "oficial" da modalidade) e `hill_climbing_fixed_budget()` (parada por orçamento fixo de avaliações, criada especificamente para comparação justa com o SA). |
| `simulated_annealing.py` | `DEFAULT_INITIAL_TEMPERATURE`; `DEFAULT_COOLING_RATE`; `SimulatedAnnealingResult` (dataclass); `simulated_annealing()` | Implementa o Simulated Annealing "puro" com resfriamento geométrico — o algoritmo principal da modalidade. |
| `simulated_annealing_reheating.py` | `DEFAULT_STAGNATION_LIMIT`; `DEFAULT_REHEAT_FACTOR`; `SimulatedAnnealingReheatingResult` (dataclass); `simulated_annealing_with_reheating()` | Implementa a variante aprimorada: idêntica ao SA puro, mas reaquece a temperatura quando a melhor solução global fica estagnada por `stagnation_limit` iterações. |
| `sa_utils.py` | `compute_cooling_rate()` | Utilitário de calibração: deriva analiticamente o `cooling_rate` que faz o resfriamento geométrico consumir o orçamento de iterações inteiro (em vez de esvaziar-se cedo demais), a partir de `initial_temperature`, `min_temperature` e `max_iterations`. |
| `experiments.py` | `run_multiple_hill_climbing()`; `ComparisonRun` (dataclass); `run_comparison_experiment()`; `ThreeWayComparisonRun` (dataclass); `run_three_way_comparison()` | Infraestrutura de execução múltipla/pareada: roda os algoritmos várias vezes com seeds derivadas de uma `seed_base`, pareando HC/SA/SA-reaquecido pela mesma seed em cada execução, para permitir testes estatísticos posteriores (Wilcoxon). |
| `__init__.py` | Docstring do pacote | Sem lógica própria — apenas identifica o pacote (`"Pacote de Busca Local: Simulated Annealing aplicado a EVRP (adaptação do GR17/TSPLIB)."`). |

## Seção 4 — Scripts na raiz de `busca-local/`

A distinção "produção vs. exploratório" segue o que cada arquivo declara
sobre si mesmo: os 5 scripts `run_official_comparison*.py` se descrevem
como scripts de experimento (reusáveis, com CSV de saída); os 4 scripts
`scratch_*.py` trazem um cabeçalho explícito "SCRIPT DESCARTÁVEL /
EXPLORATÓRIO" no topo do arquivo.

| Arquivo | Tipo | O que faz | CSV gerado |
|---|---|---|---|
| `run_official_comparison.py` | Produção | Roda `run_comparison_experiment` (HC de orçamento fixo vs. SA, schedule de resfriamento **original** `cooling_rate=0.995`), n_runs=30, Wilcoxon pareado. | `resultados/comparacao_oficial.csv` |
| `run_official_comparison_v2_schedule_corrigido.py` | Produção | Mesma comparação HC vs. SA, mas com o `cooling_rate` recalibrado por `sa_utils.compute_cooling_rate` (schedule "v2"). Rodada paralela à anterior, não substitui o CSV dela. | `resultados/comparacao_oficial_v2_schedule_corrigido.csv` |
| `run_official_comparison_three_way.py` | Produção | Roda `run_three_way_comparison` (HC vs. SA vs. SA-reaquecido), schedule de resfriamento **original** e reaquecimento com os parâmetros exploratórios **iniciais** (`stagnation_limit=500`, `reheat_factor=3.0`). | `resultados/comparacao_tres_vias.csv` |
| `run_official_comparison_three_way_v2_schedule_corrigido.py` | Produção | Mesma comparação de três vias, `cooling_rate` recalibrado (v2) para SA e SA-reaquecido — mas o reaquecimento ainda com os parâmetros exploratórios antigos (500/3.0), o que revelou a quebra do reaquecimento sob o schedule novo (1/30 execuções viáveis). | `resultados/comparacao_tres_vias_v2_schedule_corrigido.csv` |
| `run_official_comparison_three_way_v2_reaquecimento_calibrado.py` | Produção | Mesma comparação de três vias, `cooling_rate` v2 + reaquecimento **recalibrado** (`stagnation_limit=1000`, `reheat_factor=1.3`) — resultado final citado no notebook. | `resultados/comparacao_tres_vias_v2_reaquecimento_calibrado.csv` |
| `scratch_sa_pilot.py` | Exploratório/descartável | Piloto inicial: valida se a correção da penalidade (déficit de bateria) permite ao SA encontrar soluções viáveis, com hiperparâmetros de T0/alpha exploratórios, sem justificativa formal. Instrumenta `accepted_worsening_moves_after_feasibility`. | Nenhum (só imprime no terminal) |
| `scratch_hc_pilot.py` | Exploratório/descartável | Piloto equivalente ao acima, mas para Hill Climbing de orçamento fixo (mesmo total de avaliações do piloto de SA), para comparação direta HC vs. SA sob o mesmo orçamento. | Nenhum |
| `scratch_stats_check.py` | Exploratório/descartável | Reexecuta os pilotos de HC e SA acima (mesmas seeds) para calcular o Wilcoxon pareado sobre os 10 custos finais e reportar a mediana da iteração de primeira viabilidade e os contadores de piora aceita. | Nenhum |
| `scratch_cooling_diagnosis.py` | Exploratório/descartável | Não roda nenhum algoritmo de busca — só aritmética sobre a fórmula de resfriamento geométrico, comparando numericamente o schedule antigo (`cooling_rate=0.995`) com o novo (calibrado por `sa_utils.compute_cooling_rate`), confirmando que o novo consome o orçamento de iterações inteiro. | Nenhum |

## Seção 5 — Cada CSV em `resultados/`

| Arquivo | Gerado por | Linhas × colunas | Colunas | Status |
|---|---|---|---|---|
| `comparacao_oficial.csv` | `run_official_comparison.py` | 60 linhas de dados + cabeçalho (30 seeds × 2 algoritmos) × 6 colunas | `seed, algorithm, best_cost, is_feasible, first_feasible_iteration, accepted_worsening_moves_after_feasibility` | **HISTÓRICO.** Usa o schedule de resfriamento original (`cooling_rate=0.995`); mantido para rastreabilidade do processo de calibração, mas não citado no notebook para conclusões finais. |
| `comparacao_oficial_v2_schedule_corrigido.csv` | `run_official_comparison_v2_schedule_corrigido.py` | 60 × 6 (mesmas colunas do anterior) | idem | **HISTÓRICO** (etapa intermediária). Já usa o `cooling_rate` calibrado, mas é uma comparação de só DUAS vias (sem SA-reaquecido) — foi superado pela comparação de três vias final. |
| `comparacao_tres_vias.csv` | `run_official_comparison_three_way.py` | 90 linhas de dados + cabeçalho (30 seeds × 3 algoritmos) × 7 colunas | `seed, algorithm, best_cost, is_feasible, first_feasible_iteration, accepted_worsening_moves_after_feasibility, reheating_events_count` | **HISTÓRICO.** Schedule original + reaquecimento com parâmetros exploratórios iniciais (500/3.0). |
| `comparacao_tres_vias_v2_schedule_corrigido.csv` | `run_official_comparison_three_way_v2_schedule_corrigido.py` | 90 × 7 (mesmas colunas) | idem | **HISTÓRICO.** Documenta o episódio em que o reaquecimento (500/3.0) quebrou sob o schedule v2 (apenas 1/30 execuções do SA-reaquecido viáveis) — não deve ser citado para conclusões sobre desempenho, só para essa investigação específica (já discutida na Seção 6 do notebook). |
| `comparacao_tres_vias_v2_reaquecimento_calibrado.csv` | `run_official_comparison_three_way_v2_reaquecimento_calibrado.py` | 90 × 7 (mesmas colunas) | idem | **FINAL.** É a **única** fonte de dados usada na Seção 7 do notebook para a tabela comparativa, os testes de Wilcoxon e a discussão — schedule de resfriamento calibrado + reaquecimento recalibrado (1000/1.3). |

Em todas as tabelas: `seed` identifica a execução pareada (mesma seed usada
em todos os algoritmos daquela linha); `algorithm` é o nome do algoritmo
(varia entre CSVs conforme a etapa — ver os scripts na Seção 4 para o valor
exato de cada rodada); `best_cost` é o custo da melhor rota encontrada
(`problem.evaluate`); `is_feasible` indica se essa melhor rota é
efetivamente viável de bateria; `first_feasible_iteration` é a iteração em
que a primeira rota viável foi encontrada (vazio/NaN se nenhuma foi);
`accepted_worsening_moves_after_feasibility` (só SA e SA-reaquecido) conta
quantas pioras foram aceitas depois da primeira viabilidade; e
`reheating_events_count` (só nos CSVs de três vias) é o número de
reaquecimentos ocorridos naquela execução do SA-reaquecido.

## Seção 6 — Testes

| Arquivo | Nº de testes | Cobre |
|---|---|---|
| `test_dataset.py` | 9 | Carregamento e validação da matriz de distância e do tour ótimo (dimensão, simetria, diagonal zero, rejeição de dados inválidos). |
| `test_evrp_mapping.py` | 5 | Consistência das constantes de mapeamento (depósito/estações/clientes cobrem todos os 17 nós) e o comportamento de `is_feasible_route` (viabilidade simples, reset de bateria em recarga, resposta a trecho que estoura a autonomia). |
| `test_problem.py` | 6 | `route_cost`, `route_penalty` (zero quando viável, positiva e com gradiente proporcional ao déficit quando inviável) e `evaluate` como combinação das duas. |
| `test_hill_climbing.py` | 10 | `random_valid_route` (estrutura sempre válida), `hill_climbing` (execução básica, histórico de custo não-crescente, nunca piora a rota inicial), `run_multiple_hill_climbing`, `hill_climbing_fixed_budget` (respeita o orçamento exato, inclusive com divisão não-exata entre reinícios) e `run_comparison_experiment` (pareamento correto pela mesma seed). |
| `test_simulated_annealing.py` | 4 | Execução básica do SA, uso dos parâmetros de temperatura padrão, decaimento monotônico da temperatura e limites do contador de pioras aceitas após viabilidade. |
| `test_simulated_annealing_reheating.py` | 3 | Execução básica do SA com reaquecimento, ocorrência de eventos de reaquecimento quando o limiar de estagnação é baixo, e a propriedade elitista (o melhor custo global nunca piora ao longo da execução). |

**Total: 37 testes**, todos passando (`pytest -q` → `37 passed`).

## Seção 7 — Como reproduzir do zero

As instruções completas de reprodução de ambiente já estão documentadas em
`docs/DECISOES.md`, seção "Reprodução do ambiente" — não repetidas aqui
para evitar divergência. Em resumo: criar um venv, instalar
`requirements.txt` e rodar `pytest` a partir de `busca-local/`.

**Nota de atualização:** o texto de `DECISOES.md` cita apenas `numpy` e
`pytest` como pacotes fixados — isso estava correto quando foi escrito, mas
`requirements.txt` cresceu desde então (para suportar os experimentos
estatísticos e a execução do notebook). O conteúdo **atual** e completo de
`requirements.txt` (não alterado por este documento) é:

```
numpy==2.5.3
pytest==9.1.1
scipy==1.18.1
pandas==3.0.6
matplotlib==3.11.2
jupyter==1.1.1
ipykernel==7.3.0
nbclient==0.11.0
```

Para reproduzir também a execução do notebook (não só a suíte de testes),
após `pip install -r requirements.txt`:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace simulated_annealing_evrp.ipynb
```
