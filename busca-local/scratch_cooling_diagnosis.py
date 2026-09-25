# SCRIPT DESCARTAVEL / EXPLORATORIO - fora de src/, NAO faz parte da
# implementacao final do pacote busca_local. Compara o schedule de
# resfriamento ANTIGO (cooling_rate=0.995 fixo) com o schedule NOVO,
# calibrado por criterio analitico via
# `busca_local.sa_utils.compute_cooling_rate` para so atingir
# min_temperature=1.0 ao FINAL do orcamento de 20.000 iteracoes (em vez de
# esvaziar-se nos primeiros ~10% dele, como o diagnostico anterior revelou).
#
# Nao roda nenhum dos algoritmos de busca aqui - e so aritmetica sobre a
# formula de resfriamento geometrico T(n) = T0 * cooling_rate**n, para
# confirmar numericamente o comportamento dos dois schedules antes de
# reexecutar os experimentos oficiais (Passo 3).

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from busca_local.sa_utils import compute_cooling_rate

INITIAL_TEMPERATURE = 5000.0
MAX_ITERATIONS = 20_000
MIN_TEMPERATURE = 1.0

OLD_COOLING_RATE = 0.995
NEW_COOLING_RATE = compute_cooling_rate(INITIAL_TEMPERATURE, MIN_TEMPERATURE, MAX_ITERATIONS)


def temperature_at(cooling_rate, iteration):
    return INITIAL_TEMPERATURE * (cooling_rate**iteration)


def iteration_where_temperature_crosses_below(cooling_rate, threshold):
    """Primeira iteracao (inteira) em que T(n) < threshold, ou None se nunca
    (dentro de MAX_ITERATIONS) ocorrer."""
    for iteration in range(0, MAX_ITERATIONS + 1):
        if temperature_at(cooling_rate, iteration) < threshold:
            return iteration
    return None


def main():
    print(f"Parametros: initial_temperature={INITIAL_TEMPERATURE}, max_iterations={MAX_ITERATIONS}, "
          f"min_temperature_alvo={MIN_TEMPERATURE}")
    print(f"cooling_rate ANTIGO (fixo)  = {OLD_COOLING_RATE}")
    print(f"cooling_rate NOVO (calibrado) = {NEW_COOLING_RATE:.10f}")
    print()

    print(f"{'iteracao':>10} {'T (antigo)':>14} {'T (novo)':>14}")
    for iteration in range(0, MAX_ITERATIONS + 1, 1000):
        t_old = temperature_at(OLD_COOLING_RATE, iteration)
        t_new = temperature_at(NEW_COOLING_RATE, iteration)
        print(f"{iteration:>10} {t_old:>14.6f} {t_new:>14.6f}")

    print()

    cross_old = iteration_where_temperature_crosses_below(OLD_COOLING_RATE, MIN_TEMPERATURE)
    cross_new = iteration_where_temperature_crosses_below(NEW_COOLING_RATE, MIN_TEMPERATURE)

    frac_old = cross_old / MAX_ITERATIONS if cross_old is not None else None

    print(f"Schedule ANTIGO: T cruza abaixo de {MIN_TEMPERATURE} na iteracao {cross_old} "
          f"({frac_old:.4%} do orcamento de {MAX_ITERATIONS})")

    if cross_new is None:
        # Por construcao, compute_cooling_rate faz T(max_iterations) == min_temperature
        # EXATAMENTE -- T nunca fica estritamente abaixo do alvo dentro do
        # orcamento, so o atinge no ultimo passo. Isso e o resultado ESPERADO,
        # nao uma falha do diagnostico: confirma que o schedule novo usa o
        # orcamento inteiro (nao esgota o resfriamento cedo).
        print(f"Schedule NOVO:   T nunca cruza estritamente abaixo de {MIN_TEMPERATURE} dentro do orcamento "
              f"-- atinge o alvo EXATAMENTE na ultima iteracao ({MAX_ITERATIONS}/{MAX_ITERATIONS} = 100.00% do orcamento)")
        frac_new = 1.0
    else:
        frac_new = cross_new / MAX_ITERATIONS
        print(f"Schedule NOVO:   T cruza abaixo de {MIN_TEMPERATURE} na iteracao {cross_new} "
              f"({frac_new:.4%} do orcamento de {MAX_ITERATIONS})")
    print()

    print(f"T final (iteracao {MAX_ITERATIONS}) -- antigo: {temperature_at(OLD_COOLING_RATE, MAX_ITERATIONS):.6e}"
          f"   novo: {temperature_at(NEW_COOLING_RATE, MAX_ITERATIONS):.6f}")

    assert frac_new > 0.99, "schedule novo deveria consumir essencialmente o orcamento inteiro"
    print()
    print("Confirmado: o schedule novo usa o orcamento completo (so atinge min_temperature no fim),")
    print("enquanto o antigo esgota o resfriamento em uma fracao pequena do orcamento (8.5%).")


if __name__ == "__main__":
    main()
