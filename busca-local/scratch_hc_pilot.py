# SCRIPT DESCARTAVEL / EXPLORATORIO - fora de src/, NAO faz parte da
# implementacao final do pacote busca_local. Serve apenas para comparar,
# sob o MESMO orcamento total de avaliacoes usado no piloto de SA
# (scratch_sa_pilot.py: 20.000 avaliacoes por execucao, 10 execucoes), como
# o Hill Climbing se sai encontrando solucoes EVRP viaveis com a penalidade
# corrigida (gradiente de deficit de bateria).
#
# Escolha de orcamento (documentada, conforme pedido): 1 UNICO reinicio de
# exatamente 20.000 avaliacoes por execucao, SEM parar por estagnacao.
# Motivo: o hill_climbing() de producao (src/busca_local/hill_climbing.py)
# usa um CONTADOR DE ESTAGNACAO como criterio de parada por reinicio, nao um
# numero fixo de avaliacoes - "max_iterations_per_restart" la e um limiar de
# passos consecutivos sem melhora estrita, entao o total de avaliacoes
# realizadas varia (na pratica, sempre um pouco ACIMA do limiar, e reinicios
# multiplos multiplicam essa variabilidade). Isso tornaria a comparacao com
# o SA - que roda uma caminhada UNICA e continua de exatamente 20.000 passos,
# sem reinicios - injusta ou dificil de auditar.
#
# Para isolar a unica variavel experimental que interessa aqui (a REGRA DE
# ACEITACAO: HC so aceita <=, SA aceita piora com probabilidade exp(-delta/T))
# mantendo tudo o mais identico possivel - mesmos operadores de vizinhanca
# (get_neighbor), mesma funcao objetivo (evaluate, com a penalidade
# corrigida), mesmo numero de avaliacoes, e ate a MESMA rota inicial por
# execucao (mesma seed_base, primeira chamada de rng identica em ambos os
# scripts) - este piloto usa uma caminhada HC unica e continua de 20.000
# avaliacoes, sem reinicio por estagnacao e sem reinicio por orcamento
# esgotado. Multiplos reinicios menores somando 20.000 tambem seriam validos
# (e serao explorados na implementacao final), mas introduziriam uma segunda
# variavel (numero de "tentativas do zero") que confundiria esta comparacao
# pontual HC vs. SA.

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
SEED_BASE = 2026  # mesma seed_base do piloto de SA -> mesma rota inicial por execucao


def run_hc_once(distance_matrix, rng):
    """Uma caminhada HC unica e continua de N_ITERATIONS avaliacoes.

    Criterio de aceitacao identico ao de hill_climbing(): aceita o vizinho
    se seu custo for menor OU IGUAL ao custo corrente. Sem reinicio por
    estagnacao (ver justificativa no cabecalho do arquivo) - a caminhada
    roda ate esgotar o orcamento de avaliacoes, mesmo que fique presa num
    otimo local por boa parte dele.
    """
    current_route = random_valid_route(CUSTOMERS, CHARGING_STATIONS, DEPOT, rng)
    current_cost = evaluate(current_route, distance_matrix, BATTERY_CAPACITY)

    best_route = current_route
    best_cost = current_cost

    first_feasible_iteration = None
    found_feasible = is_feasible_route(current_route, distance_matrix, BATTERY_CAPACITY) is True
    if found_feasible:
        first_feasible_iteration = 0

    for iteration in range(1, N_ITERATIONS + 1):
        neighbor_route = get_neighbor(current_route, rng)
        neighbor_cost = evaluate(neighbor_route, distance_matrix, BATTERY_CAPACITY)

        if neighbor_cost <= current_cost:
            current_route, current_cost = neighbor_route, neighbor_cost

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
    }


def main():
    distance_matrix = load_distance_matrix(DISTANCE_MATRIX_PATH)

    results = []
    for i in range(N_RUNS):
        rng = random.Random(SEED_BASE + i)
        result = run_hc_once(distance_matrix, rng)
        results.append(result)

        status = "VIAVEL" if result["found_feasible"] else "inviavel"
        first_it = result["first_feasible_iteration"]
        print(
            f"run {i}: {status:8s}  primeira_viavel_iter={first_it!s:>6}  "
            f"best_cost={result['best_cost']:.1f}"
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
