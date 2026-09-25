"""Experimento oficial de comparação HC vs SA — v2, com cooling_rate recalibrado.

Rodada PARALELA ao experimento original (run_official_comparison.py), não um
substituto: mesmas seeds (seed_base=2026), mesmo n_runs=30, mesmo
total_evaluations=20000, mesmo HC (hill_climbing_fixed_budget, inalterado) —
a ÚNICA diferença é o `cooling_rate` do SA, agora calculado por
`busca_local.sa_utils.compute_cooling_rate` (ver scratch_cooling_diagnosis.py
para o diagnóstico numérico que motivou essa recalibração: o cooling_rate
antigo fixo, 0.995, esgotava o resfriamento em ~8.5% do orçamento).

Resultados salvos em resultados/comparacao_oficial_v2_schedule_corrigido.csv
(o CSV original, resultados/comparacao_oficial.csv, permanece intocado, para
comparação lado a lado no notebook).

Uso: python run_official_comparison_v2_schedule_corrigido.py
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
from busca_local.sa_utils import compute_cooling_rate

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
DISTANCE_MATRIX_PATH = DATA_DIR / "gr17_d.txt"
RESULTS_DIR = Path(__file__).resolve().parent / "resultados"
CSV_PATH = RESULTS_DIR / "comparacao_oficial_v2_schedule_corrigido.csv"

N_RUNS = 30
SEED_BASE = 2026  # mesma seed_base do experimento original
TOTAL_EVALUATIONS = 20_000

INITIAL_TEMPERATURE = 5000.0
MIN_TEMPERATURE = 1.0
NEW_COOLING_RATE = compute_cooling_rate(INITIAL_TEMPERATURE, MIN_TEMPERATURE, TOTAL_EVALUATIONS)


def summarize(label, costs):
    print(
        f"  {label}: min={min(costs):.1f}  max={max(costs):.1f}  "
        f"media={statistics.mean(costs):.1f}  mediana={statistics.median(costs):.1f}  "
        f"desvio_padrao={statistics.pstdev(costs):.1f}"
    )


def main():
    distance_matrix = load_distance_matrix(DISTANCE_MATRIX_PATH)

    print(f"cooling_rate recalibrado (T0={INITIAL_TEMPERATURE}, T_min={MIN_TEMPERATURE}, "
          f"max_iterations={TOTAL_EVALUATIONS}): {NEW_COOLING_RATE:.10f}")
    print(f"(cooling_rate original do experimento v1: 0.995)")
    print()

    comparison_runs = run_comparison_experiment(
        distance_matrix=distance_matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        n_runs=N_RUNS,
        seed_base=SEED_BASE,
        total_evaluations=TOTAL_EVALUATIONS,
        sa_kwargs={
            "initial_temperature": INITIAL_TEMPERATURE,
            "cooling_rate": NEW_COOLING_RATE,
        },
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
                "algorithm": "simulated_annealing_v2_schedule_corrigido",
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

    print(f"=== Experimento oficial v2 (schedule corrigido) — n_runs={N_RUNS}, "
          f"total_evaluations={TOTAL_EVALUATIONS} ===")
    print(f"seed_base={SEED_BASE}")
    print()

    print("Taxa de viabilidade (melhor rota encontrada e viavel):")
    print(f"  Hill Climbing (orcamento fixo):        {sum(hc_feasible_flags)}/{N_RUNS}")
    print(f"  Simulated Annealing (schedule v2):     {sum(sa_feasible_flags)}/{N_RUNS}")
    print()

    print("Distribuicao do custo final da melhor solucao (todas as 30 execucoes):")
    summarize("Hill Climbing (orcamento fixo)    ", hc_costs)
    summarize("Simulated Annealing (schedule v2) ", sa_costs)
    print()

    print("Mediana da iteracao da primeira solucao viavel (entre as execucoes que encontraram):")
    if hc_feasible_iters:
        print(
            f"  Hill Climbing (orcamento fixo):    mediana={statistics.median(hc_feasible_iters):.1f}  "
            f"(n={len(hc_feasible_iters)}/{N_RUNS})"
        )
    else:
        print("  Hill Climbing (orcamento fixo):    nenhuma execucao encontrou viabilidade")
    if sa_feasible_iters:
        print(
            f"  Simulated Annealing (schedule v2): mediana={statistics.median(sa_feasible_iters):.1f}  "
            f"(n={len(sa_feasible_iters)}/{N_RUNS})"
        )
    else:
        print("  Simulated Annealing (schedule v2): nenhuma execucao encontrou viabilidade")
    print()

    statistic, p_value = wilcoxon(hc_costs, sa_costs)
    print("Wilcoxon signed-rank pareado (custo final, HC vs SA-v2, n=30 pares):")
    print(f"  statistic={statistic:.4f}  p-value={p_value:.6f}")
    print()

    print(
        f"Total de accepted_worsening_moves_after_feasibility do SA-v2, agregado nas {N_RUNS} execucoes: "
        f"{sa_worsening_total}"
    )
    print()

    print(f"Resultados brutos salvos em: {CSV_PATH}")


if __name__ == "__main__":
    main()
