"""Carregamento e validação do dataset GR17 (TSPLIB)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DatasetValidationError(ValueError):
    """Erro levantado quando o dataset carregado não passa nas validações estruturais."""


@dataclass(frozen=True)
class OptimalTour:
    """Tour ótimo de referência do GR17.

    `one_indexed` preserva a numeração original do arquivo (1..17), útil para
    exibição/relatórios. `zero_indexed` é a mesma sequência convertida para
    índices de array (0..16), usada internamente pelos algoritmos.
    """

    one_indexed: list[int]
    zero_indexed: list[int]

    def __iter__(self):
        return iter(self.zero_indexed)

    def __len__(self) -> int:
        return len(self.zero_indexed)


def load_distance_matrix(path: str | Path) -> np.ndarray:
    """Lê a matriz de distâncias do GR17 (gr17_d.txt) e retorna um array 17x17 de inteiros.

    Levanta DatasetValidationError se a matriz não for quadrada, não for
    simétrica ou não tiver diagonal zero.
    """
    matrix = np.loadtxt(path, dtype=int)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise DatasetValidationError(
            f"Matriz de distâncias deve ser quadrada; formato lido: {matrix.shape}"
        )

    if not np.array_equal(matrix, matrix.T):
        raise DatasetValidationError(
            "Matriz de distâncias não é simétrica (matriz != matriz transposta)."
        )

    if not np.all(np.diag(matrix) == 0):
        raise DatasetValidationError(
            "Diagonal da matriz de distâncias deve ser zero (distância de um nó a ele mesmo)."
        )

    return matrix


def load_optimal_tour(path: str | Path) -> OptimalTour:
    """Lê o tour ótimo de referência do GR17 (gr17_s.txt).

    O arquivo contém um índice de cidade (1-indexado) por linha. Retorna um
    OptimalTour com a versão 1-indexada original (para exibição) e a versão
    0-indexada (para uso interno com arrays numpy).
    """
    with open(path, encoding="utf-8") as f:
        one_indexed = [int(line.strip()) for line in f if line.strip()]

    zero_indexed = [node - 1 for node in one_indexed]

    return OptimalTour(one_indexed=one_indexed, zero_indexed=zero_indexed)
