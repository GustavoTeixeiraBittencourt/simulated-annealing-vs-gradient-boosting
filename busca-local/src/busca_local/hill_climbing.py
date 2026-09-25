"""Hill Climbing com reinício aleatório para o EVRP (GR17 adaptado).

Algoritmo obrigatório da modalidade Busca Local. Usa exclusivamente a função
objetivo `problem.evaluate` e as operações de vizinhança de `neighborhood`,
para permitir comparação justa com o Simulated Annealing e a variante
aprimorada implementados posteriormente.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from busca_local.evrp_mapping import is_feasible_route
from busca_local.neighborhood import get_neighbor
from busca_local.problem import Route, evaluate, random_valid_route


@dataclass
class HillClimbingResult:
    """Resultado de uma execução completa de Hill Climbing (todos os reinícios).

    - `best_route` / `best_cost`: melhor solução encontrada entre todos os
      reinícios.
    - `cost_history`: uma lista por reinício, contendo o custo da solução
      CORRENTE (não necessariamente a melhor até então, mas equivalente a
      ela dado que o HC só aceita vizinhos melhores-ou-iguais) após cada
      iteração daquele reinício. Cada sublista é, por construção,
      monotonicamente não-crescente — usada para plotar a evolução da busca.
    - `restart_costs`: custo final alcançado em cada reinício individual
      (um valor por reinício) — usado para discutir a dependência da
      solução em relação ao estado inicial.
    - `total_iterations`: número total de avaliações de vizinhos realizadas,
      somando todos os reinícios.
    - `first_feasible_iteration`: número cumulativo de avaliações (somando
      todos os reinícios) até a primeira vez que a rota corrente se tornou
      viável, ou `None` se nenhum reinício encontrou viabilidade. Só é
      preenchido por `hill_climbing_fixed_budget`; a função `hill_climbing`
      original não rastreia isso (o campo fica `None`), para permanecer
      inalterada — daí o valor padrão `None` aqui.
    """

    best_route: Route
    best_cost: float
    cost_history: list[list[float]] = field(default_factory=list)
    restart_costs: list[float] = field(default_factory=list)
    total_iterations: int = 0
    first_feasible_iteration: int | None = None


def hill_climbing(
    distance_matrix,
    customers: list[int],
    charging_stations: list[int],
    depot: int,
    battery_capacity,
    max_iterations_per_restart: int,
    n_restarts: int,
    rng: random.Random,
) -> HillClimbingResult:
    """Hill Climbing com reinício aleatório.

    Em cada um dos `n_restarts` reinícios:
      1. Gera uma rota inicial aleatória (`random_valid_route`).
      2. A cada iteração, gera UM vizinho (`get_neighbor`) e o compara à
         solução corrente usando `evaluate`. Se o vizinho for melhor OU
         IGUAL, ele é aceito (isso permite "andar" por platôs de custo
         igual, uma variante comum de Hill Climbing).
      3. Um contador de estagnação conta iterações consecutivas SEM MELHORA
         ESTRITA (isto é, sem uma redução de custo) — passos aceitos por
         empate contam como estagnação, apenas melhora estrita zera o
         contador. Quando esse contador atinge `max_iterations_per_restart`,
         o reinício é considerado estagnado e encerrado.

    A melhor solução de cada reinício é comparada com a melhor global; a
    melhor entre todos os reinícios é retornada.
    """
    best_route: Route | None = None
    best_cost = float("inf")

    cost_history: list[list[float]] = []
    restart_costs: list[float] = []
    total_iterations = 0

    for _ in range(n_restarts):
        current_route = random_valid_route(customers, charging_stations, depot, rng)
        current_cost = evaluate(current_route, distance_matrix, battery_capacity)

        restart_history: list[float] = []
        stagnation_counter = 0

        while stagnation_counter < max_iterations_per_restart:
            neighbor_route = get_neighbor(current_route, rng)
            neighbor_cost = evaluate(neighbor_route, distance_matrix, battery_capacity)
            total_iterations += 1

            if neighbor_cost <= current_cost:
                if neighbor_cost < current_cost:
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                current_route, current_cost = neighbor_route, neighbor_cost
            else:
                stagnation_counter += 1

            restart_history.append(current_cost)

        cost_history.append(restart_history)
        restart_costs.append(current_cost)

        if current_cost < best_cost:
            best_cost = current_cost
            best_route = current_route

    return HillClimbingResult(
        best_route=best_route,
        best_cost=best_cost,
        cost_history=cost_history,
        restart_costs=restart_costs,
        total_iterations=total_iterations,
    )


def hill_climbing_fixed_budget(
    distance_matrix,
    customers: list[int],
    charging_stations: list[int],
    depot: int,
    battery_capacity,
    total_evaluations: int,
    evaluations_per_restart: int,
    rng: random.Random,
) -> HillClimbingResult:
    """Hill Climbing com reinício aleatório sob ORÇAMENTO FIXO de avaliações.

    Variante de `hill_climbing()` criada especificamente para comparação
    justa com o Simulated Annealing sob "mesmo número de avaliações"
    (`hill_climbing()` original usa um critério de ESTAGNAÇÃO como parada,
    não um número fixo — o total real de avaliações ali varia e normalmente
    excede o limiar configurado). `hill_climbing()` não é modificada; esta é
    uma função adicional e independente.

    `total_evaluations` é dividido em reinícios de tamanho
    `evaluations_per_restart` cada (o último reinício é parcial se a divisão
    não for exata). Cada reinício roda uma caminhada de Hill Climbing padrão
    (aceita vizinho se `custo <= custo corrente`) pelo número de avaliações
    alocado a ele, SEM parar por estagnação antes disso — ao contrário de
    `hill_climbing()`, aqui uma caminhada só termina por ter esgotado seu
    orçamento de avaliações. Isso garante `total_iterations == total_evaluations`
    exatamente (nunca estoura o orçamento pedido).

    A melhor solução de cada reinício é comparada com a melhor global. O
    campo `first_feasible_iteration` do `HillClimbingResult` retornado marca
    o número cumulativo de avaliações (contando todos os reinícios) até a
    primeira vez que a rota corrente se tornou viável nesta execução — ou
    `None` se nenhum reinício encontrou viabilidade.
    """
    best_route: Route | None = None
    best_cost = float("inf")

    cost_history: list[list[float]] = []
    restart_costs: list[float] = []
    total_iterations = 0
    first_feasible_iteration: int | None = None
    found_feasible = False

    remaining_evaluations = total_evaluations

    while remaining_evaluations > 0:
        this_restart_budget = min(evaluations_per_restart, remaining_evaluations)

        current_route = random_valid_route(customers, charging_stations, depot, rng)
        current_cost = evaluate(current_route, distance_matrix, battery_capacity)

        if not found_feasible and is_feasible_route(
            current_route, distance_matrix, battery_capacity
        ) is True:
            found_feasible = True
            first_feasible_iteration = total_iterations

        restart_history: list[float] = []

        for _ in range(this_restart_budget):
            neighbor_route = get_neighbor(current_route, rng)
            neighbor_cost = evaluate(neighbor_route, distance_matrix, battery_capacity)
            total_iterations += 1

            if neighbor_cost <= current_cost:
                current_route, current_cost = neighbor_route, neighbor_cost

                if not found_feasible and is_feasible_route(
                    current_route, distance_matrix, battery_capacity
                ) is True:
                    found_feasible = True
                    first_feasible_iteration = total_iterations

            restart_history.append(current_cost)

        cost_history.append(restart_history)
        restart_costs.append(current_cost)

        if current_cost < best_cost:
            best_cost = current_cost
            best_route = current_route

        remaining_evaluations -= this_restart_budget

    return HillClimbingResult(
        best_route=best_route,
        best_cost=best_cost,
        cost_history=cost_history,
        restart_costs=restart_costs,
        total_iterations=total_iterations,
        first_feasible_iteration=first_feasible_iteration,
    )
