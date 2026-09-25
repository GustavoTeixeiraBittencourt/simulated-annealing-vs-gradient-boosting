"""Mapeamento do dataset GR17 (TSP) para uma instância de EVRP.

O GR17 é originalmente um TSP puro (17 cidades, matriz de distância real,
tour ótimo conhecido). Para o módulo de Busca Local, adaptamos esse dataset
para EVRP (Electric Vehicle Routing Problem) atribuindo papéis (depósito,
estações de recarga, clientes) aos nós já existentes, sem alterar nenhuma
distância. A justificativa completa dessa decisão está em docs/DECISOES.md.

Todos os índices de nó abaixo são 1-indexados, seguindo a numeração original
do dataset GR17 (compatível com gr17_s.txt).
"""

DEPOT = 1  # 1-indexado, nó de origem/retorno
CHARGING_STATIONS = [3, 9, 14]  # 1-indexado
CUSTOMERS = [2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17]  # 1-indexado

BATTERY_CAPACITY = 800  # mesma unidade da matriz de distância (sem conversão física)
CONSUMPTION_RATE = 1.0  # unidades de bateria por unidade de distância (1:1)
RECHARGE_MODE = "full"  # recarga sempre total ao passar por estação ou depósito


def is_feasible_route(route, distance_matrix, battery_capacity=BATTERY_CAPACITY):
    """Verifica a viabilidade de bateria de uma rota EVRP.

    `route` é uma sequência de nós 1-indexados (mesma numeração de DEPOT/
    CHARGING_STATIONS/CUSTOMERS). `distance_matrix` é o array 0-indexado
    retornado por `dataset.load_distance_matrix`.

    A bateria começa no nível `battery_capacity` (assume-se partida com
    bateria cheia). A cada trecho da rota, a bateria é reduzida pela
    distância percorrida (multiplicada por CONSUMPTION_RATE). Sempre que o
    veículo chega a um nó que é o depósito ou uma estação de recarga, a
    bateria é totalmente recarregada antes do próximo trecho.

    Retorna True se a rota inteira for viável (bateria nunca fica negativa).
    Retorna (False, i) se o trecho entre route[i] e route[i+1] esgota a
    bateria antes de alcançar um ponto de recarga.
    """
    battery = float(battery_capacity)

    for i in range(len(route) - 1):
        current_node = route[i]
        next_node = route[i + 1]

        distance = distance_matrix[current_node - 1][next_node - 1]
        battery -= distance * CONSUMPTION_RATE

        if battery < 0:
            return False, i

        if next_node == DEPOT or next_node in CHARGING_STATIONS:
            battery = float(battery_capacity)

    return True


def compute_total_battery_deficit(route, distance_matrix, battery_capacity=BATTERY_CAPACITY):
    """Soma o déficit de bateria de TODOS os trechos inviáveis da rota.

    Diferente de `is_feasible_route` (que para no primeiro trecho quebrado),
    esta função continua a simulação até o fim da rota: sempre que a bateria
    ficaria negativa num trecho, o valor absoluto do déficit é somado a um
    acumulador e a bateria é resetada para 0 (nunca negativa) a partir dali,
    para poder contabilizar violações subsequentes também. Se o nó de
    chegada for o depósito ou uma estação de recarga, a bateria é
    recarregada totalmente (sobrescrevendo o reset para 0), como em
    `is_feasible_route`.

    Serve como sinal de gradiente para a função objetivo (`problem.evaluate`):
    diferencia rotas "quase viáveis" (déficit pequeno) de rotas "muito
    inviáveis" (déficit grande), ao contrário de uma penalidade em degrau.

    Retorna 0.0 se a rota for totalmente viável.
    """
    battery = float(battery_capacity)
    total_deficit = 0.0

    for i in range(len(route) - 1):
        current_node = route[i]
        next_node = route[i + 1]

        distance = distance_matrix[current_node - 1][next_node - 1]
        battery -= distance * CONSUMPTION_RATE

        if battery < 0:
            total_deficit += abs(battery)
            battery = 0.0

        if next_node == DEPOT or next_node in CHARGING_STATIONS:
            battery = float(battery_capacity)

    return total_deficit
