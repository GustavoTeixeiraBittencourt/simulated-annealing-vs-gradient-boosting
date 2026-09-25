from pathlib import Path

from busca_local.dataset import load_distance_matrix
from busca_local.evrp_mapping import BATTERY_CAPACITY, CHARGING_STATIONS, DEPOT
from busca_local.problem import (
    INFEASIBLE_PENALTY_BASE,
    evaluate,
    route_cost,
    route_penalty,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DISTANCE_MATRIX_PATH = DATA_DIR / "gr17_d.txt"


def _matrix():
    return load_distance_matrix(DISTANCE_MATRIX_PATH)


def test_route_cost_sums_consecutive_distances():
    matrix = _matrix()
    route = [DEPOT, 4, DEPOT]

    expected = matrix[DEPOT - 1][3] + matrix[3][DEPOT - 1]
    assert route_cost(route, matrix) == expected


def test_route_penalty_is_zero_for_feasible_route():
    matrix = _matrix()
    route = [DEPOT, 4, DEPOT]

    assert route_penalty(route, matrix, BATTERY_CAPACITY) == 0.0


def test_route_penalty_is_positive_for_infeasible_route():
    matrix = _matrix()
    route = [4, 5, 12]  # d(4,5)+d(5,12) = 804 > 800 (deficit = 4)

    penalty = route_penalty(route, matrix, BATTERY_CAPACITY)

    assert penalty > 0
    assert penalty == INFEASIBLE_PENALTY_BASE + 4


def test_route_penalty_has_gradient_between_small_and_large_deficit():
    """Prova que a penalidade tem gradiente, não é um degrau fixo.

    Duas rotas inviáveis, uma com déficit de bateria pequeno e outra com
    déficit grande: a penalidade da segunda deve ser maior que a da
    primeira. Isso é o que a penalidade antiga (INFEASIBLE_PENALTY + índice
    do trecho, 0-16) não conseguia expressar - duas rotas inviáveis
    ficavam quase sempre com custo ~1_000_000 independente de quão perto
    ou longe da viabilidade estivessem.
    """
    matrix = _matrix()

    small_deficit_route = [4, 5, 12]  # d(4,5)+d(5,12) = 804 -> déficit = 4
    large_deficit_route = [2, 16, 10]  # d(2,16)+d(16,10) = 1430 -> déficit = 630

    small_penalty = route_penalty(small_deficit_route, matrix, BATTERY_CAPACITY)
    large_penalty = route_penalty(large_deficit_route, matrix, BATTERY_CAPACITY)

    assert small_penalty > 0
    assert large_penalty > 0
    assert large_penalty > small_penalty


def test_evaluate_combines_cost_and_penalty():
    matrix = _matrix()

    feasible_route = [DEPOT, 4, DEPOT]
    assert evaluate(feasible_route, matrix, BATTERY_CAPACITY) == route_cost(
        feasible_route, matrix
    )

    infeasible_route = [4, 5, 12]
    expected = route_cost(infeasible_route, matrix) + route_penalty(
        infeasible_route, matrix, BATTERY_CAPACITY
    )
    assert evaluate(infeasible_route, matrix, BATTERY_CAPACITY) == expected


def test_route_penalty_recharge_resets_deficit_accumulation():
    """Uma rota que recarrega entre dois déficits não soma um déficit inflado.

    Mistura um trecho levemente inviável com uma recarga no meio - o déficit
    do segundo trecho não deve "herdar" o déficit não resolvido do primeiro
    além do que a física de bateria realmente impõe.
    """
    matrix = _matrix()
    station = CHARGING_STATIONS[0]

    route_with_recharge = [4, 5, station, 12]
    penalty = route_penalty(route_with_recharge, matrix, BATTERY_CAPACITY)

    assert penalty == 0.0
