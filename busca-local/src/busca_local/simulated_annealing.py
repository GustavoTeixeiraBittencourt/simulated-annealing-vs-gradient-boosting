"""Simulated Annealing para o EVRP (GR17 adaptado).

Algoritmo principal desta modalidade de Busca Local. Usa exclusivamente a
função objetivo `problem.evaluate` e as operações de vizinhança de
`neighborhood` — as mesmas usadas por `hill_climbing` — para permitir
comparação justa entre os dois algoritmos.

Estrutura e critério de aceitação validados por um piloto exploratório
(`scratch_sa_pilot.py`, na raiz de `busca-local/`) antes desta implementação
de produção; ver `docs/DECISOES.md` para a discussão dos resultados do
piloto (correção da penalidade de viabilidade, taxa de sucesso, etc.).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from busca_local.evrp_mapping import is_feasible_route
from busca_local.neighborhood import get_neighbor
from busca_local.problem import Route, evaluate, random_valid_route

# Valores validados no piloto exploratório (scratch_sa_pilot.py): T0 alto o
# suficiente para aceitar praticamente qualquer vizinho no início da busca
# (explorando o espaço amplamente), com resfriamento geométrico lento o
# bastante para permitir escapar de ótimos locais antes de "congelar". São
# exploratórios, não formalmente otimizados — mantidos como parâmetros (não
# hardcoded no corpo da função) justamente para permitir recalibração
# documentada mais adiante, sem alterar a assinatura da função.
DEFAULT_INITIAL_TEMPERATURE = 5000.0
DEFAULT_COOLING_RATE = 0.995


@dataclass
class SimulatedAnnealingResult:
    """Resultado de uma execução completa de Simulated Annealing.

    - `best_route` / `best_cost`: melhor solução encontrada ao longo de toda
      a execução (não necessariamente a solução corrente ao final, já que o
      SA pode aceitar passos piores).
    - `cost_history`: custo da solução CORRENTE após cada iteração (índice 0
      é o custo da rota inicial, antes de qualquer movimento) — ao contrário
      do Hill Climbing, esta série NÃO é monotonicamente não-crescente,
      porque o SA aceita pioras.
    - `temperature_history`: temperatura usada em cada iteração (índice 0 é
      `initial_temperature`, antes do primeiro resfriamento) — para plotar a
      curva de resfriamento.
    - `first_feasible_iteration`: iteração em que a rota corrente se tornou
      viável pela primeira vez, ou `None` se nunca ocorreu nesta execução.
    - `accepted_worsening_moves_after_feasibility`: quantas vezes, DEPOIS de
      atingida a primeira solução viável, o SA aceitou um vizinho de custo
      estritamente maior (validado no piloto: é essa capacidade de continuar
      aceitando pioras após a viabilidade que permite ao SA escapar de
      ótimos locais que prendem o Hill Climbing).
    """

    best_route: Route
    best_cost: float
    cost_history: list[float] = field(default_factory=list)
    temperature_history: list[float] = field(default_factory=list)
    first_feasible_iteration: int | None = None
    accepted_worsening_moves_after_feasibility: int = 0
    total_iterations: int = 0


def simulated_annealing(
    distance_matrix,
    customers: list[int],
    charging_stations: list[int],
    depot: int,
    battery_capacity,
    max_iterations: int,
    rng: random.Random,
    initial_temperature: float = DEFAULT_INITIAL_TEMPERATURE,
    cooling_rate: float = DEFAULT_COOLING_RATE,
) -> SimulatedAnnealingResult:
    """Simulated Annealing com resfriamento geométrico.

    A cada iteração:
      1. A temperatura é `T = initial_temperature * (cooling_rate ** iteracao)`
         (resfriamento geométrico).
      2. Um vizinho é gerado (`get_neighbor`) e comparado à solução corrente
         via `evaluate` (mesma função objetivo usada pelo Hill Climbing).
      3. Se o vizinho for melhor (`delta < 0`), é sempre aceito. Caso
         contrário, é aceito com probabilidade `exp(-delta / T)` — quanto
         maior a piora ou menor a temperatura, menor a chance de aceitação.

    A melhor solução encontrada ao longo de toda a execução é rastreada
    separadamente da solução corrente (que pode piorar temporariamente).
    """
    current_route = random_valid_route(customers, charging_stations, depot, rng)
    current_cost = evaluate(current_route, distance_matrix, battery_capacity)

    best_route = current_route
    best_cost = current_cost

    cost_history: list[float] = [current_cost]
    temperature_history: list[float] = [initial_temperature]

    first_feasible_iteration: int | None = None
    found_feasible = is_feasible_route(current_route, distance_matrix, battery_capacity) is True
    if found_feasible:
        first_feasible_iteration = 0

    accepted_worsening_moves_after_feasibility = 0

    for iteration in range(1, max_iterations + 1):
        was_feasible_before_this_move = found_feasible
        temperature = initial_temperature * (cooling_rate**iteration)

        neighbor_route = get_neighbor(current_route, rng)
        neighbor_cost = evaluate(neighbor_route, distance_matrix, battery_capacity)

        delta = neighbor_cost - current_cost
        if delta < 0:
            accept = True
        else:
            probability = math.exp(-delta / temperature) if temperature > 0 else 0.0
            accept = rng.random() < probability

        if accept:
            current_route, current_cost = neighbor_route, neighbor_cost

            if was_feasible_before_this_move and delta > 0:
                accepted_worsening_moves_after_feasibility += 1

            if current_cost < best_cost:
                best_cost = current_cost
                best_route = current_route

            if not found_feasible and is_feasible_route(
                current_route, distance_matrix, battery_capacity
            ) is True:
                found_feasible = True
                first_feasible_iteration = iteration

        cost_history.append(current_cost)
        temperature_history.append(temperature)

    return SimulatedAnnealingResult(
        best_route=best_route,
        best_cost=best_cost,
        cost_history=cost_history,
        temperature_history=temperature_history,
        first_feasible_iteration=first_feasible_iteration,
        accepted_worsening_moves_after_feasibility=accepted_worsening_moves_after_feasibility,
        total_iterations=max_iterations,
    )
