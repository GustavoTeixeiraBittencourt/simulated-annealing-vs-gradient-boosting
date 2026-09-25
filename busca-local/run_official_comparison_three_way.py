"""Experimento oficial de comparação de três vias: HC vs SA vs SA-reaquecido.

Script de produção (não descartável), na raiz de busca-local/. Roda
`run_three_way_comparison` (src/busca_local/experiments.py) com n_runs=30,
sob o mesmo orçamento total de avaliações para os três algoritmos, reporta
as estatísticas de comparação (incluindo os três testes de Wilcoxon
pareados) e salva os resultados brutos em
resultados/comparacao_tres_vias.csv para uso posterior no notebook.

Uso: python run_official_comparison_three_way.py
"""

import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from scipy.stats import wilcoxon

from busca_local.dataset import load_distance_matrix
from busca_local.evrp_mapping import (
    BATTERY_CAPACITY,
    CHARGING_STATIONS,
    CUSTOMERS,
    DEPOT,
    is_feasible_route,
)
from busca_local.experiments import run_three_way_comparison

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
DISTANCE_MATRIX_PATH = DATA_DIR / "gr17_d.txt"
RESULTS_DIR = Path(__file__).resolve().parent / "resultados"
CSV_PATH = RESULTS_DIR / "comparacao_tres_vias.csv"

N_RUNS = 30
SEED_BASE = 2026  # mesma seed_base dos experimentos anteriores
TOTAL_EVALUATIONS = 20_000

ALGORITHMS = ("hill_climbing_fixed_budget", "simulated_annealing", "simulated_annealing_reheating")


def summarize(label, costs):
    print(
        f"  {label:31s}: min={min(costs):.1f}  max={max(costs):.1f}  "
        f"media={statistics.mean(costs):.1f}  mediana={statistics.median(costs):.1f}  "
        f"desvio_padrao={statistics.pstdev(costs):.1f}"
    )


def main():
    distance_matrix = load_distance_matrix(DISTANCE_MATRIX_PATH)

    runs = run_three_way_comparison(
        distance_matrix=distance_matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        n_runs=N_RUNS,
        seed_base=SEED_BASE,
        total_evaluations=TOTAL_EVALUATIONS,
    )

    def result_for(run, algorithm):
        if algorithm == "hill_climbing_fixed_budget":
            return run.hc_result
        if algorithm == "simulated_annealing":
            return run.sa_result
        return run.sa_reheating_result

    rows = []
    for run in runs:
        for algorithm in ALGORITHMS:
            result = result_for(run, algorithm)
            feasible = is_feasible_route(result.best_route, distance_matrix, BATTERY_CAPACITY) is True
            rows.append(
                {
                    "seed": run.seed,
                    "algorithm": algorithm,
                    "best_cost": result.best_cost,
                    "is_feasible": feasible,
                    "first_feasible_iteration": result.first_feasible_iteration,
                    "accepted_worsening_moves_after_feasibility": getattr(
                        result, "accepted_worsening_moves_after_feasibility", ""
                    ),
                    "reheating_events_count": len(getattr(result, "reheating_events", [])) or "",
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "algorithm",
                "best_cost",
                "is_feasible",
                "first_feasible_iteration",
                "accepted_worsening_moves_after_feasibility",
                "reheating_events_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    costs = {algo: [result_for(run, algo).best_cost for run in runs] for algo in ALGORITHMS}
    feasible_flags = {
        algo: [
            is_feasible_route(result_for(run, algo).best_route, distance_matrix, BATTERY_CAPACITY) is True
            for run in runs
        ]
        for algo in ALGORITHMS
    }
    feasible_iters = {
        algo: [
            result_for(run, algo).first_feasible_iteration
            for run in runs
            if result_for(run, algo).first_feasible_iteration is not None
        ]
        for algo in ALGORITHMS
    }

    print(f"=== Comparacao de tres vias (n_runs={N_RUNS}, total_evaluations={TOTAL_EVALUATIONS}) ===")
    print(f"seed_base={SEED_BASE}")
    print()

    print("Taxa de viabilidade (melhor rota encontrada e viavel):")
    for algo in ALGORITHMS:
        print(f"  {algo:31s}: {sum(feasible_flags[algo])}/{N_RUNS}")
    print()

    print("Distribuicao do custo final da melhor solucao (todas as 30 execucoes):")
    for algo in ALGORITHMS:
        summarize(algo, costs[algo])
    print()

    print("Mediana da iteracao da primeira solucao viavel (entre as execucoes que encontraram):")
    for algo in ALGORITHMS:
        iters = feasible_iters[algo]
        if iters:
            print(f"  {algo:31s}: mediana={statistics.median(iters):.1f}  (n={len(iters)}/{N_RUNS})")
        else:
            print(f"  {algo:31s}: nenhuma execucao encontrou viabilidade")
    print()

    print("Wilcoxon signed-rank pareado (custo final, n=30 pares cada):")
    pairs = [
        ("hill_climbing_fixed_budget", "simulated_annealing"),
        ("hill_climbing_fixed_budget", "simulated_annealing_reheating"),
        ("simulated_annealing", "simulated_annealing_reheating"),
    ]
    for algo_a, algo_b in pairs:
        statistic, p_value = wilcoxon(costs[algo_a], costs[algo_b])
        print(f"  {algo_a} vs {algo_b}:")
        print(f"    statistic={statistic:.4f}  p-value={p_value:.6f}")
    print()

    sa_worsening_total = sum(run.sa_result.accepted_worsening_moves_after_feasibility for run in runs)
    sa_reheating_worsening_total = sum(
        run.sa_reheating_result.accepted_worsening_moves_after_feasibility for run in runs
    )
    reheating_events_total = sum(len(run.sa_reheating_result.reheating_events) for run in runs)
    print("Contadores agregados adicionais:")
    print(f"  SA puro: total accepted_worsening_moves_after_feasibility = {sa_worsening_total}")
    print(
        f"  SA-reaquecido: total accepted_worsening_moves_after_feasibility = "
        f"{sa_reheating_worsening_total}"
    )
    print(f"  SA-reaquecido: total de eventos de reaquecimento (soma das 30 execucoes) = {reheating_events_total}")
    print()

    print(f"Resultados brutos salvos em: {CSV_PATH}")


if __name__ == "__main__":
    main()
