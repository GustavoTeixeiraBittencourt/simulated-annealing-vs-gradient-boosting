"""Experimento oficial de comparação: Hill Climbing (orçamento fixo) vs. Simulated Annealing.

Script de produção (não descartável), na raiz de busca-local/. Roda
`run_comparison_experiment` (src/busca_local/experiments.py) com n_runs=30,
sob o mesmo orçamento total de avaliações para os dois algoritmos, reporta
as estatísticas de comparação (incluindo o teste de Wilcoxon pareado) e
salva os resultados brutos em resultados/comparacao_oficial.csv para uso
posterior no notebook.

Uso: python run_official_comparison.py
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
from busca_local.experiments import run_comparison_experiment

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
DISTANCE_MATRIX_PATH = DATA_DIR / "gr17_d.txt"
RESULTS_DIR = Path(__file__).resolve().parent / "resultados"
CSV_PATH = RESULTS_DIR / "comparacao_oficial.csv"

N_RUNS = 30
SEED_BASE = 2026  # mesma seed_base dos pilotos exploratorios (scratch_*_pilot.py)
TOTAL_EVALUATIONS = 20_000


def summarize(label, costs):
    print(
        f"  {label}: min={min(costs):.1f}  max={max(costs):.1f}  "
        f"media={statistics.mean(costs):.1f}  mediana={statistics.median(costs):.1f}  "
        f"desvio_padrao={statistics.pstdev(costs):.1f}"
    )


def main():
    distance_matrix = load_distance_matrix(DISTANCE_MATRIX_PATH)

    comparison_runs = run_comparison_experiment(
        distance_matrix=distance_matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        n_runs=N_RUNS,
        seed_base=SEED_BASE,
        total_evaluations=TOTAL_EVALUATIONS,
    )

    rows = []
    for run in comparison_runs:
        hc_feasible = is_feasible_route(run.hc_result.best_route, distance_matrix, BATTERY_CAPACITY) is True
        sa_feasible = is_feasible_route(run.sa_result.best_route, distance_matrix, BATTERY_CAPACITY) is True

        rows.append(
            {
                "seed": run.seed,
                "algorithm": "hill_climbing_fixed_budget",
                "best_cost": run.hc_result.best_cost,
                "is_feasible": hc_feasible,
                "first_feasible_iteration": run.hc_result.first_feasible_iteration,
                "accepted_worsening_moves_after_feasibility": "",
            }
        )
        rows.append(
            {
                "seed": run.seed,
                "algorithm": "simulated_annealing",
                "best_cost": run.sa_result.best_cost,
                "is_feasible": sa_feasible,
                "first_feasible_iteration": run.sa_result.first_feasible_iteration,
                "accepted_worsening_moves_after_feasibility": (
                    run.sa_result.accepted_worsening_moves_after_feasibility
                ),
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    hc_costs = [run.hc_result.best_cost for run in comparison_runs]
    sa_costs = [run.sa_result.best_cost for run in comparison_runs]
    hc_feasible_flags = [
        is_feasible_route(run.hc_result.best_route, distance_matrix, BATTERY_CAPACITY) is True
        for run in comparison_runs
    ]
    sa_feasible_flags = [
        is_feasible_route(run.sa_result.best_route, distance_matrix, BATTERY_CAPACITY) is True
        for run in comparison_runs
    ]
    hc_feasible_iters = [
        run.hc_result.first_feasible_iteration
        for run in comparison_runs
        if run.hc_result.first_feasible_iteration is not None
    ]
    sa_feasible_iters = [
        run.sa_result.first_feasible_iteration
        for run in comparison_runs
        if run.sa_result.first_feasible_iteration is not None
    ]
    sa_worsening_total = sum(
        run.sa_result.accepted_worsening_moves_after_feasibility for run in comparison_runs
    )

    print(f"=== Experimento oficial de comparacao (n_runs={N_RUNS}, total_evaluations={TOTAL_EVALUATIONS}) ===")
    print(f"seed_base={SEED_BASE}")
    print()

    print("Taxa de viabilidade (melhor rota encontrada e viavel):")
    print(f"  Hill Climbing (orcamento fixo): {sum(hc_feasible_flags)}/{N_RUNS}")
    print(f"  Simulated Annealing:            {sum(sa_feasible_flags)}/{N_RUNS}")
    print()

    print("Distribuicao do custo final da melhor solucao (todas as 30 execucoes):")
    summarize("Hill Climbing (orcamento fixo)", hc_costs)
    summarize("Simulated Annealing           ", sa_costs)
    print()

    print("Mediana da iteracao da primeira solucao viavel (entre as execucoes que encontraram):")
    if hc_feasible_iters:
        print(
            f"  Hill Climbing (orcamento fixo): mediana={statistics.median(hc_feasible_iters):.1f}  "
            f"(n={len(hc_feasible_iters)}/{N_RUNS})"
        )
    else:
        print("  Hill Climbing (orcamento fixo): nenhuma execucao encontrou viabilidade")
    if sa_feasible_iters:
        print(
            f"  Simulated Annealing:            mediana={statistics.median(sa_feasible_iters):.1f}  "
            f"(n={len(sa_feasible_iters)}/{N_RUNS})"
        )
    else:
        print("  Simulated Annealing:            nenhuma execucao encontrou viabilidade")
    print()

    statistic, p_value = wilcoxon(hc_costs, sa_costs)
    print("Wilcoxon signed-rank pareado (custo final, HC vs SA, n=30 pares):")
    print(f"  statistic={statistic:.4f}  p-value={p_value:.6f}")
    print()

    print(
        f"Total de accepted_worsening_moves_after_feasibility do SA, agregado nas {N_RUNS} execucoes: "
        f"{sa_worsening_total}"
    )
    print()

    print(f"Resultados brutos salvos em: {CSV_PATH}")


if __name__ == "__main__":
    main()
