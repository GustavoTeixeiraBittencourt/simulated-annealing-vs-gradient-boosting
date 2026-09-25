"""Execução múltipla de Hill Climbing e Simulated Annealing para análise estatística.

Suporta a exigência do edital de rodar os algoritmos "múltiplas vezes sob
condições comparáveis" e reportar melhor/pior/média/taxa de sucesso.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from busca_local.hill_climbing import HillClimbingResult, hill_climbing, hill_climbing_fixed_budget
from busca_local.simulated_annealing import SimulatedAnnealingResult, simulated_annealing
from busca_local.simulated_annealing_reheating import (
    SimulatedAnnealingReheatingResult,
    simulated_annealing_with_reheating,
)


def run_multiple_hill_climbing(
    n_runs: int,
    seed_base: int,
    **hc_kwargs,
) -> list[HillClimbingResult]:
    """Roda `hill_climbing` `n_runs` vezes, uma seed diferente por execução.

    As seeds são derivadas de `seed_base` como `seed_base + i` para
    `i` em `range(n_runs)`, garantindo execuções reprodutíveis e
    independentes entre si. `hc_kwargs` deve conter todos os argumentos de
    `hill_climbing` exceto `rng` (que é criado aqui a partir da seed
    derivada): `distance_matrix`, `customers`, `charging_stations`, `depot`,
    `battery_capacity`, `max_iterations_per_restart`, `n_restarts`.

    Retorna a lista de `HillClimbingResult`, um por execução, na mesma
    ordem das seeds.
    """
    results: list[HillClimbingResult] = []

    for i in range(n_runs):
        rng = random.Random(seed_base + i)
        result = hill_climbing(rng=rng, **hc_kwargs)
        results.append(result)

    return results


@dataclass
class ComparisonRun:
    """Par de resultados (HC de orçamento fixo vs. SA) de UMA execução pareada.

    `seed` é a seed usada para gerar o `random.Random` de AMBOS os
    algoritmos nesta execução (instâncias separadas, mas com o mesmo valor
    de seed) — o que faz os dois partirem da mesma rota inicial, isolando a
    regra de aceitação/busca como a variável de comparação.
    """

    seed: int
    hc_result: HillClimbingResult
    sa_result: SimulatedAnnealingResult


def run_comparison_experiment(
    distance_matrix,
    customers: list[int],
    charging_stations: list[int],
    depot: int,
    battery_capacity,
    n_runs: int = 30,
    seed_base: int = 0,
    total_evaluations: int = 20_000,
    evaluations_per_restart: int | None = None,
    sa_kwargs: dict | None = None,
    hc_kwargs: dict | None = None,
) -> list[ComparisonRun]:
    """Roda Hill Climbing (orçamento fixo) e Simulated Annealing pareados.

    Para cada uma das `n_runs` execuções (seeds `seed_base + i`), roda
    `hill_climbing_fixed_budget` e `simulated_annealing` com o MESMO
    `total_evaluations` e a MESMA seed (instâncias distintas de
    `random.Random(seed)`, uma para cada algoritmo) — garantindo que ambos
    partem da mesma rota inicial em cada execução pareada.

    `evaluations_per_restart` (para o HC) tem como padrão `None`, que
    significa "um único reinício de `total_evaluations` avaliações" (isto é,
    `evaluations_per_restart = total_evaluations`). Essa é a mesma escolha
    documentada e validada em `scratch_hc_pilot.py`: uma caminhada HC única e
    contínua, sem reinícios, para isolar a regra de aceitação como a única
    variável de comparação com o SA (que também roda uma caminhada única).
    Passe um valor explícito para usar múltiplos reinícios menores.

    `sa_kwargs` / `hc_kwargs` são dicionários opcionais com argumentos extras
    repassados a `simulated_annealing` (ex.: `initial_temperature`,
    `cooling_rate`) e `hill_climbing_fixed_budget`, respectivamente.

    Retorna a lista de `ComparisonRun`, uma por execução (não agregada), para
    permitir teste estatístico pareado posterior (ex.: Wilcoxon).
    """
    if evaluations_per_restart is None:
        evaluations_per_restart = total_evaluations

    sa_kwargs = dict(sa_kwargs) if sa_kwargs else {}
    hc_kwargs = dict(hc_kwargs) if hc_kwargs else {}

    results: list[ComparisonRun] = []

    for i in range(n_runs):
        seed = seed_base + i

        hc_result = hill_climbing_fixed_budget(
            distance_matrix=distance_matrix,
            customers=customers,
            charging_stations=charging_stations,
            depot=depot,
            battery_capacity=battery_capacity,
            total_evaluations=total_evaluations,
            evaluations_per_restart=evaluations_per_restart,
            rng=random.Random(seed),
            **hc_kwargs,
        )

        sa_result = simulated_annealing(
            distance_matrix=distance_matrix,
            customers=customers,
            charging_stations=charging_stations,
            depot=depot,
            battery_capacity=battery_capacity,
            max_iterations=total_evaluations,
            rng=random.Random(seed),
            **sa_kwargs,
        )

        results.append(ComparisonRun(seed=seed, hc_result=hc_result, sa_result=sa_result))

    return results


@dataclass
class ThreeWayComparisonRun:
    """Trio de resultados (HC de orçamento fixo, SA puro, SA com reaquecimento)
    de UMA execução pareada — mesma seed usada nos três algoritmos."""

    seed: int
    hc_result: HillClimbingResult
    sa_result: SimulatedAnnealingResult
    sa_reheating_result: SimulatedAnnealingReheatingResult


def run_three_way_comparison(
    distance_matrix,
    customers: list[int],
    charging_stations: list[int],
    depot: int,
    battery_capacity,
    n_runs: int = 30,
    seed_base: int = 0,
    total_evaluations: int = 20_000,
    evaluations_per_restart: int | None = None,
    sa_kwargs: dict | None = None,
    hc_kwargs: dict | None = None,
    sa_reheating_kwargs: dict | None = None,
) -> list[ThreeWayComparisonRun]:
    """Roda HC (orçamento fixo), SA puro e SA com reaquecimento pareados.

    Mesma lógica de pareamento de `run_comparison_experiment` (mesma seed,
    via `random.Random(seed)` independente por algoritmo, para os três
    partirem da mesma rota inicial em cada execução), estendida para os três
    algoritmos desta modalidade. `evaluations_per_restart=None` (padrão)
    também significa "um único reinício de `total_evaluations`" para o HC,
    pela mesma razão documentada em `run_comparison_experiment`.

    `sa_kwargs`, `hc_kwargs` e `sa_reheating_kwargs` são dicionários opcionais
    com argumentos extras repassados a `simulated_annealing`,
    `hill_climbing_fixed_budget` e `simulated_annealing_with_reheating`,
    respectivamente (ex.: `stagnation_limit`/`reheat_factor` para o último).

    Retorna a lista de `ThreeWayComparisonRun`, uma por execução (não
    agregada), para permitir testes estatísticos pareados posteriores.
    """
    if evaluations_per_restart is None:
        evaluations_per_restart = total_evaluations

    sa_kwargs = dict(sa_kwargs) if sa_kwargs else {}
    hc_kwargs = dict(hc_kwargs) if hc_kwargs else {}
    sa_reheating_kwargs = dict(sa_reheating_kwargs) if sa_reheating_kwargs else {}

    results: list[ThreeWayComparisonRun] = []

    for i in range(n_runs):
        seed = seed_base + i

        hc_result = hill_climbing_fixed_budget(
            distance_matrix=distance_matrix,
            customers=customers,
            charging_stations=charging_stations,
            depot=depot,
            battery_capacity=battery_capacity,
            total_evaluations=total_evaluations,
            evaluations_per_restart=evaluations_per_restart,
            rng=random.Random(seed),
            **hc_kwargs,
        )

        sa_result = simulated_annealing(
            distance_matrix=distance_matrix,
            customers=customers,
            charging_stations=charging_stations,
            depot=depot,
            battery_capacity=battery_capacity,
            max_iterations=total_evaluations,
            rng=random.Random(seed),
            **sa_kwargs,
        )

        sa_reheating_result = simulated_annealing_with_reheating(
            distance_matrix=distance_matrix,
            customers=customers,
            charging_stations=charging_stations,
            depot=depot,
            battery_capacity=battery_capacity,
            max_iterations=total_evaluations,
            rng=random.Random(seed),
            **sa_reheating_kwargs,
        )

        results.append(
            ThreeWayComparisonRun(
                seed=seed,
                hc_result=hc_result,
                sa_result=sa_result,
                sa_reheating_result=sa_reheating_result,
            )
        )

    return results
