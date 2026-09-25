"""Comparação de três vias — v2, com cooling_rate recalibrado (HC vs SA vs SA-reaquecido).

Rodada PARALELA à comparação de três vias original
(run_official_comparison_three_way.py), não um substituto: mesmas seeds
(seed_base=2026), mesmo n_runs=30, mesmo total_evaluations=20000, mesmo HC
(hill_climbing_fixed_budget, inalterado) — a diferença é o `cooling_rate` de
AMBOS SA e SA-reaquecido, agora calculado por
`busca_local.sa_utils.compute_cooling_rate` (ver
run_official_comparison_v2_schedule_corrigido.py: com esse schedule, o SA
puro já bateu o HC de forma nitidamente significativa, p=0.000055 — condição
para rodar este script, conforme combinado).

`stagnation_limit` e `reheat_factor` do SA-reaquecido permanecem nos valores
exploratórios padrão (500 e 3.0) — não foram recalibrados aqui, só o
cooling_rate compartilhado com o SA puro. A pergunta que este script
responde é: com um schedule que já não esfria cedo demais, o reaquecimento
ainda tem algum efeito mensurável sobre o SA puro, ou se torna redundante?

Resultados salvos em resultados/comparacao_tres_vias_v2_schedule_corrigido.csv
(o CSV original, resultados/comparacao_tres_vias.csv, permanece intocado).

Uso: python run_official_comparison_three_way_v2_schedule_corrigido.py
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
from busca_local.sa_utils import compute_cooling_rate

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
DISTANCE_MATRIX_PATH = DATA_DIR / "gr17_d.txt"
RESULTS_DIR = Path(__file__).resolve().parent / "resultados"
CSV_PATH = RESULTS_DIR / "comparacao_tres_vias_v2_schedule_corrigido.csv"

N_RUNS = 30
SEED_BASE = 2026
TOTAL_EVALUATIONS = 20_000

INITIAL_TEMPERATURE = 5000.0
MIN_TEMPERATURE = 1.0
NEW_COOLING_RATE = compute_cooling_rate(INITIAL_TEMPERATURE, MIN_TEMPERATURE, TOTAL_EVALUATIONS)

ALGORITHMS = (
    "hill_climbing_fixed_budget",
    "simulated_annealing_v2",
    "simulated_annealing_reheating_v2",
)


def summarize(label, costs):
    print(
        f"  {label:33s}: min={min(costs):.1f}  max={max(costs):.1f}  "
        f"media={statistics.mean(costs):.1f}  mediana={statistics.median(costs):.1f}  "
        f"desvio_padrao={statistics.pstdev(costs):.1f}"
    )


def main():
    distance_matrix = load_distance_matrix(DISTANCE_MATRIX_PATH)

    print(f"cooling_rate recalibrado (T0={INITIAL_TEMPERATURE}, T_min={MIN_TEMPERATURE}, "
          f"max_iterations={TOTAL_EVALUATIONS}): {NEW_COOLING_RATE:.10f}  (usado por SA e SA-reaquecido)")
    print()

    runs = run_three_way_comparison(
        distance_matrix=distance_matrix,
        customers=CUSTOMERS,
        charging_stations=CHARGING_STATIONS,
        depot=DEPOT,
        battery_capacity=BATTERY_CAPACITY,
        n_runs=N_RUNS,
        seed_base=SEED_BASE,
        total_evaluations=TOTAL_EVALUATIONS,
        sa_kwargs={"initial_temperature": INITIAL_TEMPERATURE, "cooling_rate": NEW_COOLING_RATE},
        sa_reheating_kwargs={"initial_temperature": INITIAL_TEMPERATURE, "cooling_rate": NEW_COOLING_RATE},
    )

    def result_for(run, algorithm):
        if algorithm == "hill_climbing_fixed_budget":
            return run.hc_result
        if algorithm == "simulated_annealing_v2":
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

    print(f"=== Comparacao de tres vias v2 (schedule corrigido) — n_runs={N_RUNS}, "
          f"total_evaluations={TOTAL_EVALUATIONS} ===")
    print(f"seed_base={SEED_BASE}")
    print()

    print("Taxa de viabilidade (melhor rota encontrada e viavel):")
    for algo in ALGORITHMS:
        print(f"  {algo:33s}: {sum(feasible_flags[algo])}/{N_RUNS}")
    print()

    print("Distribuicao do custo final da melhor solucao (todas as 30 execucoes):")
    for algo in ALGORITHMS:
        summarize(algo, costs[algo])
    print()

    print("Mediana da iteracao da primeira solucao viavel (entre as execucoes que encontraram):")
    for algo in ALGORITHMS:
        iters = feasible_iters[algo]
        if iters:
            print(f"  {algo:33s}: mediana={statistics.median(iters):.1f}  (n={len(iters)}/{N_RUNS})")
        else:
            print(f"  {algo:33s}: nenhuma execucao encontrou viabilidade")
    print()

    print("Wilcoxon signed-rank pareado (custo final, n=30 pares cada):")
    pairs = [
        ("hill_climbing_fixed_budget", "simulated_annealing_v2"),
        ("hill_climbing_fixed_budget", "simulated_annealing_reheating_v2"),
        ("simulated_annealing_v2", "simulated_annealing_reheating_v2"),
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
    print(f"  SA-v2: total accepted_worsening_moves_after_feasibility = {sa_worsening_total}")
    print(f"  SA-reaquecido-v2: total accepted_worsening_moves_after_feasibility = {sa_reheating_worsening_total}")
    print(f"  SA-reaquecido-v2: total de eventos de reaquecimento (soma das 30 execucoes) = {reheating_events_total}")
    print()

    print(f"Resultados brutos salvos em: {CSV_PATH}")


if __name__ == "__main__":
    main()
