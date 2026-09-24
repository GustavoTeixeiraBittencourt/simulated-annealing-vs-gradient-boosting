"""Configuração central do experimento: sementes, caminhos, meta de acurácia e grades de busca."""

from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
DIR_DADOS_RAW = RAIZ_PROJETO / "data" / "raw"
DIR_RESULTADOS_FIGURAS = RAIZ_PROJETO / "resultados" / "figuras"
DIR_RESULTADOS_TABELAS = RAIZ_PROJETO / "resultados" / "tabelas"

SEMENTE_PRINCIPAL = 42
SEMENTES_ESTABILIDADE = list(range(10))

PROPORCAO_TESTE = 0.2
DOBRAS_VALIDACAO_CRUZADA = 5

# Definida antes da busca de hiperparâmetros, acima das baselines medidas em
# avaliacao.py (~70%) e dentro do teto empírico observado (~72%) — ver docs/DECISOES.md.
META_ACURACIA = 0.71

GRADE_PROFUNDIDADE_MAXIMA = [1, 2, 3, 4, 5, 6, 7, 8, None]
GRADE_MIN_AMOSTRAS_FOLHA = [1, 5, 10, 20, 40]

# Grade menor: a chefboost expande um ramo por categoria sem poda real, então
# o treino fica caro rápido demais em profundidades maiores (ver docs/DECISOES.md).
GRADE_PROFUNDIDADE_MAXIMA_C45 = [1, 2, 3, 4, 5, 6]

GB_N_ESTIMATORS = 100
GB_LEARNING_RATE = 0.1
GB_MAX_DEPTH = 3

OPENML_NOME_DATASET = "credit-g"
OPENML_VERSAO_DATASET = 1
COLUNA_ALVO = "class"
ROTULO_POSITIVO = "bad"
