import random
from pathlib import Path

from busca_local.dataset import load_distance_matrix
from busca_local.evrp_mapping import BATTERY_CAPACITY, CHARGING_STATIONS, CUSTOMERS, DEPOT
from busca_local.simulated_annealing_reheating import simulated_annealing_with_reheating

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DISTANCE_MATRIX_PATH = DATA_DIR / "gr17_d.txt"


def _matrix():
    return load_distance_matrix(DISTANCE_MATRIX_PATH)


def test_runs_without_error_and_visits_each_customer_once():
    matrix = _matrix()
    rng = random.Random(7)

    result = simulated_annealing_with_reheating(
        distance_matrix=matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        max_iterations=500,
        rng=rng,
    )

    assert result.best_route[0] == DEPOT
    assert result.best_route[-1] == DEPOT
    visited_customers = [node for node in result.best_route if node in CUSTOMERS]
    assert sorted(visited_customers) == sorted(CUSTOMERS)
    assert result.total_iterations == 500
    assert len(result.cost_history) == 501
    assert len(result.temperature_history) == 501


def test_reheating_events_occur_when_stagnation_limit_is_low_enough():
    """Com um stagnation_limit baixo o suficiente para o orcamento de teste,
    pelo menos um reaquecimento deve ocorrer (a busca converge/estagna bem
    antes de 2000 iteracoes neste problema, entao um limite de 50 iteracoes
    consecutivas sem melhora do melhor global e quase certo de ser atingido).
    """
    matrix = _matrix()
    rng = random.Random(5)

    result = simulated_annealing_with_reheating(
        distance_matrix=matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        max_iterations=2000,
        rng=rng,
        stagnation_limit=50,
        reheat_factor=3.0,
    )

    assert len(result.reheating_events) > 0
    # cada evento marca uma iteracao valida dentro do orcamento da execucao
    assert all(1 <= iteration <= result.total_iterations for iteration in result.reheating_events)


def test_global_best_never_regresses():
    """Propriedade elitista: mesmo aceitando pioras localmente (SA), o
    'melhor até agora' (`best_cost`) reportado ao final nunca pode ser pior
    do que o melhor custo já visitado em QUALQUER ponto da execução — ou
    seja, `best_cost` deve ser exatamente o mínimo de `cost_history` (que
    contém o custo da solução corrente a cada iteração, incluindo a
    inicial). Se o bookkeeping de melhor global regredisse ou perdesse uma
    melhora, `best_cost` seria maior que esse mínimo.
    """
    matrix = _matrix()
    rng = random.Random(11)

    result = simulated_annealing_with_reheating(
        distance_matrix=matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        max_iterations=1500,
        rng=rng,
        stagnation_limit=100,
        reheat_factor=3.0,
    )

    assert result.best_cost == min(result.cost_history)
