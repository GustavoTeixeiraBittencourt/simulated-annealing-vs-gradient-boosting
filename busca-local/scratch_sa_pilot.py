# SCRIPT DESCARTÁVEL / EXPLORATÓRIO — fora de src/, NÃO faz parte da
# implementação final do pacote busca_local. Serve apenas para validar, de
# forma rápida, se a correção da penalidade (route_penalty baseada em
# déficit de bateria) permite que uma busca que aceita passos piores
# (Simulated Annealing) encontre soluções EVRP viáveis. Os hiperparâmetros
# de SA usados aqui (T0, alpha) são exploratórios, escolhidos sem
# justificativa formal — a implementação final de SA (fora deste script)
# fará a calibração e a justificativa apropriadas.

import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from busca_local.dataset import load_distance_matrix
from busca_local.evrp_mapping import (
    BATTERY_CAPACITY,
    CHARGING_STATIONS,
    CUSTOMERS,
    DEPOT,
    is_feasible_route,
)
from busca_local.neighborhood import get_neighbor
from busca_local.problem import evaluate, random_valid_route

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
DISTANCE_MATRIX_PATH = DATA_DIR / "gr17_d.txt"

N_ITERATIONS = 20_000
N_RUNS = 10
SEED_BASE = 2026
T0 = 5000.0
ALPHA = 0.995


def run_sa_once(distance_matrix, rng):
    """Uma execução de SA exploratório. Retorna um dicionário com métricas."""
    current_route = random_valid_route(CUSTOMERS, CHARGING_STATIONS, DEPOT, rng)
    current_cost = evaluate(current_route, distance_matrix, BATTERY_CAPACITY)

    best_route = current_route
    best_cost = current_cost

    first_feasible_iteration = None
    found_feasible = is_feasible_route(current_route, distance_matrix, BATTERY_CAPACITY) is True
    if found_feasible:
        first_feasible_iteration = 0

    # Conta quantas vezes, DEPOIS de já ter atingido a primeira solução
    # viável, o SA aceitou um vizinho estritamente pior (delta > 0). Não
    # conta a própria jogada que alcançou a viabilidade (nesse instante
    # `found_feasible` ainda é False no início da iteração), nem empates
    # (delta == 0, que não são "piora").
    accepted_worsening_moves_after_feasibility = 0

    for iteration in range(1, N_ITERATIONS + 1):
        was_feasible_before_this_move = found_feasible
        temperature = T0 * (ALPHA ** iteration)

        neighbor_route = get_neighbor(current_route, rng)
        neighbor_cost = evaluate(neighbor_route, distance_matrix, BATTERY_CAPACITY)

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
                current_route, distance_matrix, BATTERY_CAPACITY
            ) is True:
                found_feasible = True
                first_feasible_iteration = iteration

    return {
        "found_feasible": found_feasible,
        "first_feasible_iteration": first_feasible_iteration,
        "best_cost": best_cost,
        "best_route": best_route,
        "accepted_worsening_moves_after_feasibility": accepted_worsening_moves_after_feasibility,
    }


def main():
    distance_matrix = load_distance_matrix(DISTANCE_MATRIX_PATH)

    results = []
    for i in range(N_RUNS):
        rng = random.Random(SEED_BASE + i)
        result = run_sa_once(distance_matrix, rng)
        results.append(result)

        status = "VIAVEL" if result["found_feasible"] else "inviavel"
        first_it = result["first_feasible_iteration"]
        worsening = result["accepted_worsening_moves_after_feasibility"]
        print(
            f"run {i}: {status:8s}  primeira_viavel_iter={first_it!s:>6}  "
            f"best_cost={result['best_cost']:.1f}  "
            f"pioras_aceitas_pos_viabilidade={worsening}"
        )

    print()
    print("=== Resumo agregado ===")

    n_feasible_runs = sum(1 for r in results if r["found_feasible"])
    print(f"Execucoes que encontraram viabilidade: {n_feasible_runs}/{N_RUNS}")

    feasible_iters = [
        r["first_feasible_iteration"] for r in results if r["found_feasible"]
    ]
    if feasible_iters:
        mean_it = statistics.mean(feasible_iters)
        std_it = statistics.pstdev(feasible_iters) if len(feasible_iters) > 1 else 0.0
        median_it = statistics.median(feasible_iters)
        print(
            f"Iteracao da primeira solucao viavel (entre as que encontraram): "
            f"media={mean_it:.1f} (desvio padrao {std_it:.1f})  mediana={median_it:.1f}"
        )
    else:
        print("Nenhuma execucao encontrou viabilidade — sem iteracao media a reportar.")

    worsening_counts = [r["accepted_worsening_moves_after_feasibility"] for r in results]
    print(
        "Pioras aceitas apos a primeira viabilidade, por execucao: "
        + ", ".join(f"run{i}={c}" for i, c in enumerate(worsening_counts))
    )
    print(
        f"  total={sum(worsening_counts)}  media={statistics.mean(worsening_counts):.1f}  "
        f"mediana={statistics.median(worsening_counts):.1f}"
    )

    feasible_run_costs = [r["best_cost"] for r in results if r["found_feasible"]]
    infeasible_run_costs = [r["best_cost"] for r in results if not r["found_feasible"]]

    def describe(label, costs):
        if not costs:
            print(f"  {label}: nenhuma execucao nessa categoria")
            return
        print(
            f"  {label} (n={len(costs)}): min={min(costs):.1f} "
            f"max={max(costs):.1f} media={statistics.mean(costs):.1f}"
        )

    print("Distribuicao de custo final da melhor solucao de cada execucao:")
    describe("Execucoes VIAVEIS", feasible_run_costs)
    describe("Execucoes inviaveis", infeasible_run_costs)


if __name__ == "__main__":
    main()
