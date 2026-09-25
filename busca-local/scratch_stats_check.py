# SCRIPT DESCARTAVEL / EXPLORATORIO - fora de src/, NAO faz parte da
# implementacao final do pacote busca_local. Reexecuta os pilotos de HC e
# SA (scratch_hc_pilot.py / scratch_sa_pilot.py) run a run, com as MESMAS
# seeds (SEED_BASE + i), para obter os 10 pares HC-vs-SA por indice de
# execucao e:
#
#   1) rodar scipy.stats.wilcoxon PAREADO sobre os 10 custos finais
#      (HC[i] vs SA[i], mesma seed em cada par);
#   2) reportar a MEDIANA (nao so media/desvio) da iteracao da primeira
#      solucao viavel, para os dois algoritmos;
#   3) reportar, por execucao de SA, accepted_worsening_moves_after_feasibility
#      (ja instrumentado em scratch_sa_pilot.py).
#
# Reexecutar aqui (em vez de so reler numeros ja impressos) garante que os
# tres resultados vêm exatamente do mesmo par de rodadas, sem risco de
# comparar execucoes de momentos diferentes.

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from scipy.stats import wilcoxon

import scratch_hc_pilot as hc_pilot
import scratch_sa_pilot as sa_pilot
from busca_local.dataset import load_distance_matrix

assert hc_pilot.SEED_BASE == sa_pilot.SEED_BASE, "pilotos devem usar a mesma SEED_BASE para parear por seed"
assert hc_pilot.N_RUNS == sa_pilot.N_RUNS == 10, "esta analise assume 10 execucoes pareadas"

SEED_BASE = hc_pilot.SEED_BASE
N_RUNS = hc_pilot.N_RUNS


def main():
    distance_matrix = load_distance_matrix(hc_pilot.DISTANCE_MATRIX_PATH)

    hc_results = []
    sa_results = []
    for i in range(N_RUNS):
        hc_results.append(hc_pilot.run_hc_once(distance_matrix, random.Random(SEED_BASE + i)))
        sa_results.append(sa_pilot.run_sa_once(distance_matrix, random.Random(SEED_BASE + i)))

    hc_costs = [r["best_cost"] for r in hc_results]
    sa_costs = [r["best_cost"] for r in sa_results]

    print("=== 1) Wilcoxon pareado (custo final HC vs SA, por seed) ===")
    print(f"{'seed':>6} {'HC_cost':>10} {'SA_cost':>10} {'diff(HC-SA)':>12}")
    for i in range(N_RUNS):
        diff = hc_costs[i] - sa_costs[i]
        print(f"{SEED_BASE + i:>6} {hc_costs[i]:>10.1f} {sa_costs[i]:>10.1f} {diff:>12.1f}")

    statistic, p_value = wilcoxon(hc_costs, sa_costs)
    print(f"\nWilcoxon signed-rank: statistic={statistic:.4f}  p-value={p_value:.6f}")

    print()
    print("=== 2) Mediana da iteracao da primeira solucao viavel ===")

    def feasible_iters(results):
        return [r["first_feasible_iteration"] for r in results if r["found_feasible"]]

    for label, results in (("Hill Climbing", hc_results), ("Simulated Annealing", sa_results)):
        iters = feasible_iters(results)
        n_feasible = len(iters)
        if iters:
            mean_it = statistics.mean(iters)
            std_it = statistics.pstdev(iters) if len(iters) > 1 else 0.0
            median_it = statistics.median(iters)
            print(
                f"{label:22s}: n_viaveis={n_feasible}/{N_RUNS}  media={mean_it:.1f}  "
                f"desvio_padrao={std_it:.1f}  MEDIANA={median_it:.1f}"
            )
        else:
            print(f"{label:22s}: nenhuma execucao encontrou viabilidade")

    print()
    print("=== 3) Pioras aceitas pelo SA apos a primeira viabilidade (por execucao) ===")
    worsening_counts = [r["accepted_worsening_moves_after_feasibility"] for r in sa_results]
    for i, count in enumerate(worsening_counts):
        print(f"  seed={SEED_BASE + i}: accepted_worsening_moves_after_feasibility={count}")
    print(
        f"  total={sum(worsening_counts)}  media={statistics.mean(worsening_counts):.1f}  "
        f"mediana={statistics.median(worsening_counts):.1f}"
    )


if __name__ == "__main__":
    main()
