"""Representação do problema EVRP: rotas, função objetivo e geração de rotas iniciais.

Esta é a ÚNICA função objetivo (`evaluate`) usada por todos os algoritmos de
Busca Local da modalidade (Hill Climbing, Simulated Annealing e a variante
aprimorada). Nenhum outro algoritmo deve implementar seu próprio cálculo de
custo — isso é o que torna a comparação entre eles válida.
"""

from __future__ import annotations

import random

from busca_local.evrp_mapping import compute_total_battery_deficit

Route = list[int]

# Penalidade base aplicada a QUALQUER rota inviável, somada ao déficit total
# de bateria (ver `compute_total_battery_deficit`). Uma ordem de grandeza
# (1000) acima do custo típico de uma rota viável no GR17 (~2000-3000 de
# distância), o suficiente para que nenhuma rota inviável seja preferida a
# uma rota viável de custo comparável — mas SEM apagar o sinal de distância:
# diferente da penalidade em degrau anterior (INFEASIBLE_PENALTY = 1e6 fixo
# + índice do trecho quebrado, 0-16), aqui o déficit de bateria entra
# proporcionalmente na penalidade, dando à busca um gradiente real entre
# "quase viável" (déficit pequeno) e "muito inviável" (déficit grande).
INFEASIBLE_PENALTY_BASE = 1000

# Probabilidade de inserir uma estação de recarga logo após cada cliente ao
# gerar uma rota aleatória (ver `random_valid_route`). Escolhida
# empiricamente: alta o suficiente para que uma fração razoável das rotas
# geradas já nasça viável (dado BATTERY_CAPACITY=800 e a distância total do
# tour ótimo ~2085, cerca de duas recargas são necessárias), mas baixa o
# suficiente para não inflar demais o tamanho da rota com estações
# redundantes.
STATION_INSERTION_PROBABILITY = 0.3


def route_cost(route: Route, distance_matrix) -> float:
    """Soma das distâncias entre nós consecutivos da rota (1-indexados)."""
    return float(
        sum(
            distance_matrix[route[i] - 1][route[i + 1] - 1]
            for i in range(len(route) - 1)
        )
    )


def route_penalty(route: Route, distance_matrix, battery_capacity) -> float:
    """Penalidade de viabilidade de bateria da rota, proporcional ao déficit.

    Retorna 0.0 se a rota for totalmente viável (déficit de bateria zero).
    Caso contrário, retorna INFEASIBLE_PENALTY_BASE somado ao déficit total
    de bateria (ver `compute_total_battery_deficit`) — o que dá à busca um
    gradiente real entre rotas "quase viáveis" (déficit pequeno) e "muito
    inviáveis" (déficit grande), em vez de um valor em degrau que trata
    todas as rotas inviáveis como igualmente ruins.
    """
    deficit = compute_total_battery_deficit(route, distance_matrix, battery_capacity)

    if deficit <= 0:
        return 0.0

    return INFEASIBLE_PENALTY_BASE + deficit


def evaluate(route: Route, distance_matrix, battery_capacity) -> float:
    """Função objetivo única: custo de distância + penalidade de viabilidade.

    Usada por Hill Climbing, Simulated Annealing e a variante aprimorada —
    não deve ser reimplementada ou substituída por nenhum desses algoritmos.
    """
    return route_cost(route, distance_matrix) + route_penalty(
        route, distance_matrix, battery_capacity
    )


def random_valid_route(
    customers: list[int],
    charging_stations: list[int],
    depot: int,
    rng: random.Random,
    station_insertion_probability: float = STATION_INSERTION_PROBABILITY,
) -> Route:
    """Gera uma rota inicial aleatória sintaticamente válida.

    "Válida" aqui significa apenas: começa e termina no depósito, e visita
    cada cliente em `customers` exatamente uma vez — a estrutura exigida de
    uma rota EVRP. NÃO há garantia de viabilidade de bateria (isso é
    verificado separadamente por `is_feasible_route` / `route_penalty`).

    Critério de inserção de estações (decisão de design): os clientes são
    embaralhados aleatoriamente e, após CADA cliente inserido na rota, com
    probabilidade `station_insertion_probability` uma estação de recarga
    aleatória (de `charging_stations`) é inserida logo em seguida. Isso não
    garante viabilidade, mas aumenta bastante a chance de a rota já nascer
    viável (ou perto disso), evitando que quase todas as rotas iniciais
    sejam trivialmente inviáveis e a busca gaste as primeiras iterações só
    corrigindo bateria. Rotas que nascem inviáveis são penalizadas por
    `evaluate` e podem ser corrigidas ao longo das iterações de Hill
    Climbing / Simulated Annealing via `insert_station_neighbor`.
    """
    shuffled_customers = list(customers)
    rng.shuffle(shuffled_customers)

    route: Route = [depot]
    for customer in shuffled_customers:
        route.append(customer)
        if rng.random() < station_insertion_probability:
            route.append(rng.choice(charging_stations))
    route.append(depot)

    return route
