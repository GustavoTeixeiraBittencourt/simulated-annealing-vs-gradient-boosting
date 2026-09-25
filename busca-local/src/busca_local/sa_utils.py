"""Utilitários de calibração para o Simulated Annealing (SA e SA-reaquecido).

Módulo separado de `simulated_annealing.py` e `simulated_annealing_reheating.py`
de propósito: esses dois módulos continuam intocados (permanecem os
algoritmos "oficiais" já validados) — este é um utilitário de CALIBRAÇÃO de
parâmetros, usado para derivar um `cooling_rate` a partir de um critério
analítico, em vez de escolhido por tentativa e erro.
"""

from __future__ import annotations


def compute_cooling_rate(
    initial_temperature: float, min_temperature: float, max_iterations: int
) -> float:
    """Calcula a taxa de resfriamento geométrico que ocupa o orçamento inteiro.

    O resfriamento geométrico usado pelo SA é `T(n) = T0 * cooling_rate ** n`.
    Dado um orçamento fixo de `max_iterations` avaliações, queremos que a
    temperatura só atinja `min_temperature` exatamente ao FINAL desse
    orçamento — não muito antes. Resolvendo `T0 * cooling_rate ** max_iterations
    = min_temperature` para `cooling_rate`:

        cooling_rate = (min_temperature / initial_temperature) ** (1 / max_iterations)

    Por que isso importa (achado do diagnóstico anterior, ver
    `docs/DECISOES.md`): com `cooling_rate=0.995` fixo e `initial_temperature=5000`,
    a temperatura já cruza abaixo de 1.0 por volta da iteração ~1900 — menos
    de 10% de um orçamento de 20.000 avaliações. A partir daí, a
    probabilidade de aceitar um vizinho pior (`exp(-delta/T)`) fica
    desprezível para deltas de custo típicos (dezenas a centenas), e o SA
    passa a se comportar como uma busca gulosa comum pelo resto da execução
    — inclusive tornando o reaquecimento (`simulated_annealing_reheating`)
    quase sem efeito, já que um `reheat_factor` moderado não é suficiente
    para reabrir a exploração a partir de uma temperatura já tão baixa.
    Calibrar `cooling_rate` para esvaziar-se só ao FIM do orçamento faz o SA
    explorar de forma probabilística durante toda a execução, não só no
    início.
    """
    return (min_temperature / initial_temperature) ** (1 / max_iterations)
