import random
from pathlib import Path

from busca_local.dataset import load_distance_matrix
from busca_local.evrp_mapping import BATTERY_CAPACITY, CHARGING_STATIONS, CUSTOMERS, DEPOT
from busca_local.simulated_annealing import simulated_annealing

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DISTANCE_MATRIX_PATH = DATA_DIR / "gr17_d.txt"


def _matrix():
    return load_distance_matrix(DISTANCE_MATRIX_PATH)


def test_simulated_annealing_runs_without_error_and_visits_each_customer_once():
    matrix = _matrix()
    rng = random.Random(7)

    result = simulated_annealing(
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
    assert len(result.cost_history) == 501  # inclui o custo inicial (iteracao 0)
    assert len(result.temperature_history) == 501


def test_simulated_annealing_uses_default_temperature_parameters_when_omitted():
    matrix = _matrix()
    rng = random.Random(7)

    result = simulated_annealing(
        distance_matrix=matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        max_iterations=10,
        rng=rng,
    )

    assert result.temperature_history[0] == 5000.0


def test_temperature_decays_monotonically():
    matrix = _matrix()
    rng = random.Random(3)

    result = simulated_annealing(
        distance_matrix=matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        max_iterations=300,
        rng=rng,
        initial_temperature=5000.0,
        cooling_rate=0.995,
    )

    temperatures = result.temperature_history
    for earlier, later in zip(temperatures, temperatures[1:]):
        assert later < earlier


def test_accepted_worsening_moves_after_feasibility_is_non_negative_and_bounded():
    matrix = _matrix()
    rng = random.Random(2026)  # seed conhecida de scratch_sa_pilot.py que atinge viabilidade

    result = simulated_annealing(
        distance_matrix=matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        max_iterations=20_000,
        rng=rng,
    )

    assert result.first_feasible_iteration is not None
    assert 0 <= result.accepted_worsening_moves_after_feasibility <= result.total_iterations
