"""Operações de vizinhança para rotas EVRP, reusadas por HC, SA e a variante.

Todas as funções recebem a rota atual e um `random.Random` e retornam uma
NOVA rota (a rota original nunca é modificada in-place).
"""

from __future__ import annotations

import random

from busca_local.evrp_mapping import CHARGING_STATIONS, DEPOT
from busca_local.problem import Route


def swap_neighbor(route: Route, rng: random.Random) -> Route:
    """Troca de posição dois nós não-depósito escolhidos aleatoriamente.

    Se a rota tiver menos de dois nós não-depósito, retorna uma cópia
    inalterada da rota (não-operação).
    """
    new_route = list(route)

    non_depot_indices = [i for i, node in enumerate(new_route) if node != DEPOT]
    if len(non_depot_indices) < 2:
        return new_route

    i, j = rng.sample(non_depot_indices, 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def insert_station_neighbor(route: Route, rng: random.Random) -> Route:
    """Insere uma estação de recarga aleatória em uma posição interna aleatória.

    A posição de inserção nunca é o início (índice 0) nem o fim da rota, de
    modo que o depósito permanece no início e no fim. Se a rota for curta
    demais para ter uma posição interna (menos de 3 nós), retorna uma cópia
    inalterada.
    """
    new_route = list(route)

    if len(new_route) < 3:
        return new_route

    station = rng.choice(CHARGING_STATIONS)
    position = rng.randint(1, len(new_route) - 1)
    new_route.insert(position, station)
    return new_route


def remove_station_neighbor(route: Route, rng: random.Random) -> Route:
    """Remove uma estação de recarga presente na rota, se houver alguma.

    Como nenhuma estação é obrigatória na rota (ver `problem.random_valid_route`),
    qualquer ocorrência de um nó em CHARGING_STATIONS é elegível para remoção.
    Se não houver nenhuma estação na rota, retorna uma cópia inalterada
    (não-operação).
    """
    new_route = list(route)

    station_indices = [i for i, node in enumerate(new_route) if node in CHARGING_STATIONS]
    if not station_indices:
        return new_route

    index_to_remove = rng.choice(station_indices)
    del new_route[index_to_remove]
    return new_route


def get_neighbor(route: Route, rng: random.Random) -> Route:
    """Escolhe uniformemente uma das três operações de vizinhança e a aplica.

    Note que `insert_station_neighbor` e `remove_station_neighbor` podem ser
    não-operações em rotas degeneradas (respectivamente: rota muito curta, ou
    rota sem nenhuma estação) — isso é aceitável no contexto de Hill
    Climbing/Simulated Annealing, onde um vizinho idêntico à rota atual
    simplesmente não traz melhora naquela iteração.
    """
    operation = rng.choice([swap_neighbor, insert_station_neighbor, remove_station_neighbor])
    return operation(route, rng)
