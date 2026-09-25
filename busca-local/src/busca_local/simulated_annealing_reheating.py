"""Simulated Annealing com reaquecimento — a variante aprimorada do EVRP.

Terceiro algoritmo obrigatório desta modalidade de Busca Local. NÃO
substitui `simulated_annealing` (a versão "pura" continua sendo um dos três
algoritmos comparados) — é uma variante que adiciona reaquecimento: quando a
melhor solução global fica estagnada por `stagnation_limit` iterações
consecutivas, a temperatura corrente é multiplicada por `reheat_factor`,
permitindo à busca voltar a explorar mais amplamente mesmo depois de já ter
esfriado bastante (algo que o resfriamento geométrico puro não permite, já
que a temperatura ali só decresce).

Mesma função objetivo (`problem.evaluate`) e mesmos operadores de vizinhança
(`neighborhood.get_neighbor`) que `hill_climbing` e `simulated_annealing`,
para manter a comparação entre os três algoritmos válida.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from busca_local.evrp_mapping import is_feasible_route
from busca_local.neighborhood import get_neighbor
from busca_local.problem import Route, evaluate, random_valid_route
from busca_local.simulated_annealing import DEFAULT_COOLING_RATE, DEFAULT_INITIAL_TEMPERATURE

# Valores recalibrados (ver docs/DECISOES.md para o histórico completo):
# a primeira escolha exploratória foi stagnation_limit=500 / reheat_factor=3.0,
# validada apenas contra o cooling_rate=0.995 fixo. Quando o cooling_rate foi
# recalibrado por critério analítico (ver `sa_utils.compute_cooling_rate`,
# que mantém a temperatura alta por muito mais tempo, usando o orçamento
# inteiro em vez de esfriar em ~8.5% dele), essa combinação quebrou: um
# reheat_factor=3.0 aplicado a uma temperatura ainda alta (milhares, não
# frações de 1) dispara `exp(-delta/T) ≈ 1` para qualquer delta, tornando a
# busca essencialmente um passeio aleatório e destruindo o progresso já
# feito (medido: 1/30 execuções viáveis, custo médio mais que dobrado frente
# ao SA puro). A combinação abaixo é mais conservadora nas duas frentes:
# stagnation_limit=1000 (reaquece com menos frequência, dando mais tempo
# para a busca convergir antes de perturbá-la de novo) e reheat_factor=1.3
# (um "empurrão" de 30% na temperatura, não uma triplicação) — pensada para
# não sobrepor a exploração natural do schedule já lento. Continuam sendo
# parâmetros com default, não hardcoded no corpo da função, para permitir
# recalibração futura sem alterar a assinatura.
DEFAULT_STAGNATION_LIMIT = 1000
DEFAULT_REHEAT_FACTOR = 1.3


@dataclass
class SimulatedAnnealingReheatingResult:
    """Resultado de uma execução de Simulated Annealing com reaquecimento.

    Mesma estrutura de `SimulatedAnnealingResult` (ver docstring lá para os
    campos compartilhados), acrescida de:

    - `reheating_events`: lista das iterações em que um reaquecimento
      ocorreu (contador de estagnação do melhor global atingiu
      `stagnation_limit`) — para marcar esses pontos no gráfico de
      temperatura por iteração.
    """

    best_route: Route
    best_cost: float
    cost_history: list[float] = field(default_factory=list)
    temperature_history: list[float] = field(default_factory=list)
    first_feasible_iteration: int | None = None
    accepted_worsening_moves_after_feasibility: int = 0
    total_iterations: int = 0
    reheating_events: list[int] = field(default_factory=list)


def simulated_annealing_with_reheating(
    distance_matrix,
    customers: list[int],
    charging_stations: list[int],
    depot: int,
    battery_capacity,
    max_iterations: int,
    rng: random.Random,
    initial_temperature: float = DEFAULT_INITIAL_TEMPERATURE,
    cooling_rate: float = DEFAULT_COOLING_RATE,
    stagnation_limit: int = DEFAULT_STAGNATION_LIMIT,
    reheat_factor: float = DEFAULT_REHEAT_FACTOR,
) -> SimulatedAnnealingReheatingResult:
    """Simulated Annealing com resfriamento geométrico + reaquecimento.

    Nota sobre a ordem dos parâmetros: por exigência da sintaxe do Python
    (parâmetros com valor padrão não podem vir antes de parâmetros
    obrigatórios na mesma lista), `initial_temperature`, `cooling_rate`,
    `stagnation_limit` e `reheat_factor` — todos com default — foram
    colocados APÓS `max_iterations` e `rng` (obrigatórios, sem default),
    mesma convenção já usada em `simulated_annealing`. Chame sempre por
    nome (`stagnation_limit=...`) para evitar depender dessa ordem.

    Idêntico a `simulated_annealing()` em tudo (mesmo resfriamento
    geométrico `T *= cooling_rate` a cada iteração, mesma regra de
    aceitação `exp(-delta / T)`), exceto por um mecanismo adicional:

    Um contador de estagnação acompanha há quantas iterações CONSECUTIVAS a
    melhor solução GLOBAL (`best_cost`) não melhora. Sempre que esse contador
    atinge `stagnation_limit`, a temperatura CORRENTE é multiplicada por
    `reheat_factor` (reaquecimento) e o contador é zerado — permitindo à
    busca escapar de um ótimo local mesmo depois de a temperatura já ter
    esfriado bastante.
    """
    current_route = random_valid_route(customers, charging_stations, depot, rng)
    current_cost = evaluate(current_route, distance_matrix, battery_capacity)

    best_route = current_route
    best_cost = current_cost

    cost_history: list[float] = [current_cost]
    temperature_history: list[float] = [initial_temperature]
    reheating_events: list[int] = []

    first_feasible_iteration: int | None = None
    found_feasible = is_feasible_route(current_route, distance_matrix, battery_capacity) is True
    if found_feasible:
        first_feasible_iteration = 0

    accepted_worsening_moves_after_feasibility = 0
    stagnation_counter = 0
    temperature = initial_temperature

    for iteration in range(1, max_iterations + 1):
        was_feasible_before_this_move = found_feasible
        temperature *= cooling_rate

        neighbor_route = get_neighbor(current_route, rng)
        neighbor_cost = evaluate(neighbor_route, distance_matrix, battery_capacity)

        delta = neighbor_cost - current_cost
        if delta < 0:
            accept = True
        else:
            probability = math.exp(-delta / temperature) if temperature > 0 else 0.0
            accept = rng.random() < probability

        improved_global_best = False
        if accept:
            current_route, current_cost = neighbor_route, neighbor_cost

            if was_feasible_before_this_move and delta > 0:
                accepted_worsening_moves_after_feasibility += 1

            if current_cost < best_cost:
                best_cost = current_cost
                best_route = current_route
                improved_global_best = True

            if not found_feasible and is_feasible_route(
                current_route, distance_matrix, battery_capacity
            ) is True:
                found_feasible = True
                first_feasible_iteration = iteration

        if improved_global_best:
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= stagnation_limit:
            temperature *= reheat_factor
            reheating_events.append(iteration)
            stagnation_counter = 0

        cost_history.append(current_cost)
        temperature_history.append(temperature)

    return SimulatedAnnealingReheatingResult(
        best_route=best_route,
        best_cost=best_cost,
        cost_history=cost_history,
        temperature_history=temperature_history,
        first_feasible_iteration=first_feasible_iteration,
        accepted_worsening_moves_after_feasibility=accepted_worsening_moves_after_feasibility,
        total_iterations=max_iterations,
        reheating_events=reheating_events,
    )
