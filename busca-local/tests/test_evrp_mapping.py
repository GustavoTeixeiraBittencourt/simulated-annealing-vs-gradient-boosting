from pathlib import Path

from busca_local.dataset import load_distance_matrix
from busca_local.evrp_mapping import (
    BATTERY_CAPACITY,
    CHARGING_STATIONS,
    CUSTOMERS,
    DEPOT,
    is_feasible_route,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DISTANCE_MATRIX_PATH = DATA_DIR / "gr17_d.txt"


def test_constants_are_disjoint_and_cover_all_nodes():
    all_nodes = {DEPOT, *CHARGING_STATIONS, *CUSTOMERS}
    assert all_nodes == set(range(1, 18))
    assert len({DEPOT}) + len(CHARGING_STATIONS) + len(CUSTOMERS) == 17


def test_feasible_route_through_depot_at_start_and_end():
    matrix = load_distance_matrix(DISTANCE_MATRIX_PATH)

    route = [DEPOT, 4, DEPOT]

    assert is_feasible_route(route, matrix) is True


def test_infeasible_route_with_segment_exceeding_battery_capacity():
    matrix = load_distance_matrix(DISTANCE_MATRIX_PATH)

    route = [4, 5, 12]
    assert matrix[3][4] + matrix[4][11] > BATTERY_CAPACITY
    assert 5 not in ({DEPOT, *CHARGING_STATIONS})

    result = is_feasible_route(route, matrix)

    assert result == (False, 1)


def test_feasible_route_recharges_at_charging_station():
    matrix = load_distance_matrix(DISTANCE_MATRIX_PATH)

    station = CHARGING_STATIONS[0]
    route = [DEPOT, station, DEPOT]

    assert is_feasible_route(route, matrix) is True


def test_recharge_resets_battery_between_segments_not_total_sum():
    """Prova que a checagem é por segmento entre recargas, não soma acumulada.

    Rota cliente(2) -> estação(9) -> cliente(10): d(2,9)=555, d(9,10)=495.
    A soma total (1050) excede BATTERY_CAPACITY (800), mas cada perna
    isolada cabe na autonomia. Uma implementação que apenas somasse a
    distância total e comparasse contra 800 reprovaria essa rota
    incorretamente; a implementação correta reseta a bateria ao passar
    pela estação (nó 9) e deve aprová-la.
    """
    matrix = load_distance_matrix(DISTANCE_MATRIX_PATH)

    station = 9
    assert station in CHARGING_STATIONS

    route = [2, station, 10]
    leg1 = matrix[route[0] - 1][route[1] - 1]
    leg2 = matrix[route[1] - 1][route[2] - 1]

    assert leg1 <= BATTERY_CAPACITY
    assert leg2 <= BATTERY_CAPACITY
    assert leg1 + leg2 > BATTERY_CAPACITY  # a soma "ingênua" falharia

    assert is_feasible_route(route, matrix) is True
