import random
from pathlib import Path

from busca_local.dataset import load_distance_matrix
from busca_local.evrp_mapping import (
    BATTERY_CAPACITY,
    CHARGING_STATIONS,
    CUSTOMERS,
    DEPOT,
    is_feasible_route,
)
from busca_local.experiments import run_comparison_experiment, run_multiple_hill_climbing
from busca_local.hill_climbing import hill_climbing, hill_climbing_fixed_budget
from busca_local.problem import evaluate, random_valid_route
from busca_local.simulated_annealing import simulated_annealing

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DISTANCE_MATRIX_PATH = DATA_DIR / "gr17_d.txt"


def _matrix():
    return load_distance_matrix(DISTANCE_MATRIX_PATH)


# --- random_valid_route: estrutura sempre válida; viabilidade de bateria NÃO é garantida ---
#
# Comportamento esperado e documentado (ver docstring de `random_valid_route`):
# a rota gerada SEMPRE tem estrutura válida (depósito nas pontas, cada cliente
# uma vez), mas raramente nasce viável em termos de bateria: sandwichar um
# cliente entre dois pontos de recarga custa a SOMA das duas pernas (a
# recarga só ocorre ao CHEGAR num depósito/estação, não ao passar por um
# cliente), e essa soma facilmente excede BATTERY_CAPACITY=800 dado que a
# distância média de aresta no GR17 é ~275.
#
# NOTA HISTÓRICA (corrigida): com a penalidade antiga (degrau fixo
# INFEASIBLE_PENALTY=1e6 + índice do trecho quebrado), o Hill Climbing NUNCA
# conseguia eliminar essa inviabilidade em milhares de iterações — não havia
# gradiente entre "quase viável" e "muito inviável". Isso foi corrigido
# substituindo a penalidade por INFEASIBLE_PENALTY_BASE + déficit total de
# bateria (`compute_total_battery_deficit`, ver `problem.route_penalty`).
# Com a penalidade corrigida, o HC passou a encontrar soluções viáveis em
# 20/20 execuções de teste (ver docs/DECISOES.md). O teste abaixo
# (`test_hill_climbing_never_worsens_initial_route`) mantém apenas a
# propriedade estrutural básica que sempre foi verdadeira: o HC nunca piora
# a solução inicial de um reinício.
def test_random_valid_route_structure_is_always_valid():
    rng = random.Random(42)
    route = random_valid_route(CUSTOMERS, CHARGING_STATIONS, DEPOT, rng)

    assert route[0] == DEPOT
    assert route[-1] == DEPOT
    visited_customers = [node for node in route if node in CUSTOMERS]
    assert sorted(visited_customers) == sorted(CUSTOMERS)


def test_random_valid_route_feasibility_is_not_guaranteed_but_well_defined():
    matrix = _matrix()
    outcomes = []
    for seed in range(30):
        rng = random.Random(seed)
        route = random_valid_route(CUSTOMERS, CHARGING_STATIONS, DEPOT, rng)
        outcomes.append(is_feasible_route(route, matrix, BATTERY_CAPACITY))

    # Cada resultado é `True` ou uma tupla `(False, idx)` bem formada -
    # nunca uma exceção ou um valor inesperado.
    for outcome in outcomes:
        assert outcome is True or (
            isinstance(outcome, tuple) and outcome[0] is False and isinstance(outcome[1], int)
        )


def test_hill_climbing_runs_without_error_and_visits_each_customer_once():
    matrix = _matrix()
    rng = random.Random(7)

    result = hill_climbing(
        distance_matrix=matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        max_iterations_per_restart=20,
        n_restarts=1,
        rng=rng,
    )

    assert result.best_route[0] == DEPOT
    assert result.best_route[-1] == DEPOT
    visited_customers = [node for node in result.best_route if node in CUSTOMERS]
    assert sorted(visited_customers) == sorted(CUSTOMERS)
    assert result.total_iterations > 0
    assert len(result.restart_costs) == 1


def test_cost_history_is_non_increasing_within_each_restart():
    matrix = _matrix()
    rng = random.Random(11)

    result = hill_climbing(
        distance_matrix=matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        max_iterations_per_restart=50,
        n_restarts=3,
        rng=rng,
    )

    assert len(result.cost_history) == 3
    for restart_history in result.cost_history:
        for earlier, later in zip(restart_history, restart_history[1:]):
            assert later <= earlier


def test_hill_climbing_never_worsens_initial_route():
    """O HC nunca piora a solução inicial de um reinício (aceita só <=).

    Propriedade estrutural garantida pelo critério de aceitação do HC,
    independentemente de a rota inicial ser viável ou não.
    """
    matrix = _matrix()
    seed = 0

    route_rng = random.Random(seed)
    initial_route = random_valid_route(CUSTOMERS, CHARGING_STATIONS, DEPOT, route_rng)
    initial_cost = evaluate(initial_route, matrix, BATTERY_CAPACITY)

    hc_rng = random.Random(seed)
    result = hill_climbing(
        distance_matrix=matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        max_iterations_per_restart=300,
        n_restarts=1,
        rng=hc_rng,
    )

    assert result.best_cost <= initial_cost


def test_run_multiple_hill_climbing_returns_n_runs_results_with_variation():
    matrix = _matrix()

    results = run_multiple_hill_climbing(
        n_runs=5,
        seed_base=100,
        distance_matrix=matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        max_iterations_per_restart=30,
        n_restarts=2,
    )

    assert len(results) == 5

    best_costs = [r.best_cost for r in results]
    best_routes = [tuple(r.best_route) for r in results]
    assert len(set(best_costs)) > 1 or len(set(best_routes)) > 1


# --- hill_climbing_fixed_budget: reinício por ORÇAMENTO FIXO, para comparação com o SA ---


def test_hill_climbing_fixed_budget_runs_without_error_and_visits_each_customer_once():
    matrix = _matrix()
    rng = random.Random(9)

    result = hill_climbing_fixed_budget(
        distance_matrix=matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        total_evaluations=500,
        evaluations_per_restart=500,
        rng=rng,
    )

    assert result.best_route[0] == DEPOT
    assert result.best_route[-1] == DEPOT
    visited_customers = [node for node in result.best_route if node in CUSTOMERS]
    assert sorted(visited_customers) == sorted(CUSTOMERS)


def test_hill_climbing_fixed_budget_respects_total_evaluations_single_restart():
    matrix = _matrix()
    rng = random.Random(9)

    result = hill_climbing_fixed_budget(
        distance_matrix=matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        total_evaluations=1000,
        evaluations_per_restart=1000,
        rng=rng,
    )

    assert result.total_iterations == 1000


def test_hill_climbing_fixed_budget_respects_total_evaluations_with_uneven_split():
    """total_evaluations que nao divide exatamente evaluations_per_restart:
    o ultimo reinicio deve ser parcial, mas o total nunca deve estourar."""
    matrix = _matrix()
    rng = random.Random(9)

    result = hill_climbing_fixed_budget(
        distance_matrix=matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        total_evaluations=1050,
        evaluations_per_restart=300,
        rng=rng,
    )

    assert result.total_iterations == 1050
    assert len(result.cost_history) == 4  # 300 + 300 + 300 + 150
    assert [len(restart) for restart in result.cost_history] == [300, 300, 300, 150]


# --- run_comparison_experiment: HC (orcamento fixo) vs SA, pareados pela mesma seed ---


def test_run_comparison_experiment_pairs_same_seed_for_both_algorithms():
    matrix = _matrix()
    n_runs = 3
    seed_base = 500
    total_evaluations = 200

    comparison_results = run_comparison_experiment(
        distance_matrix=matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        n_runs=n_runs,
        seed_base=seed_base,
        total_evaluations=total_evaluations,
    )

    assert len(comparison_results) == n_runs

    for i, run in enumerate(comparison_results):
        expected_seed = seed_base + i
        assert run.seed == expected_seed

        # Reproduzir cada algoritmo isoladamente com a MESMA seed deve dar
        # exatamente o mesmo resultado que run_comparison_experiment produziu
        # -- prova de que a mesma seed foi de fato usada para os dois.
        expected_hc = hill_climbing_fixed_budget(
            distance_matrix=matrix,
            customers=CUSTOMERS,
            charging_stations=CHARGING_STATIONS,
            depot=DEPOT,
            battery_capacity=BATTERY_CAPACITY,
            total_evaluations=total_evaluations,
            evaluations_per_restart=total_evaluations,
            rng=random.Random(expected_seed),
        )
        assert run.hc_result.best_cost == expected_hc.best_cost
        assert run.hc_result.best_route == expected_hc.best_route

        expected_sa = simulated_annealing(
            distance_matrix=matrix,
            customers=CUSTOMERS,
            charging_stations=CHARGING_STATIONS,
            depot=DEPOT,
            battery_capacity=BATTERY_CAPACITY,
            max_iterations=total_evaluations,
            rng=random.Random(expected_seed),
        )
        assert run.sa_result.best_cost == expected_sa.best_cost
        assert run.sa_result.best_route == expected_sa.best_route
