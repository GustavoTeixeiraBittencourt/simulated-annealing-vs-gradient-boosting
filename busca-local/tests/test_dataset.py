from pathlib import Path

import numpy as np
import pytest

from busca_local.dataset import DatasetValidationError, load_distance_matrix, load_optimal_tour

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DISTANCE_MATRIX_PATH = DATA_DIR / "gr17_d.txt"
OPTIMAL_TOUR_PATH = DATA_DIR / "gr17_s.txt"


def test_distance_matrix_is_17x17():
    matrix = load_distance_matrix(DISTANCE_MATRIX_PATH)
    assert matrix.shape == (17, 17)


def test_distance_matrix_is_symmetric():
    matrix = load_distance_matrix(DISTANCE_MATRIX_PATH)
    assert np.array_equal(matrix, matrix.T)


def test_distance_matrix_diagonal_is_zero():
    matrix = load_distance_matrix(DISTANCE_MATRIX_PATH)
    assert np.all(np.diag(matrix) == 0)


def test_load_distance_matrix_rejects_asymmetric_matrix(tmp_path):
    bad_matrix_path = tmp_path / "bad_asymmetric.txt"
    bad_matrix_path.write_text("0 1 2\n1 0 3\n2 4 0\n")

    with pytest.raises(DatasetValidationError):
        load_distance_matrix(bad_matrix_path)


def test_load_distance_matrix_rejects_nonzero_diagonal(tmp_path):
    bad_matrix_path = tmp_path / "bad_diagonal.txt"
    bad_matrix_path.write_text("1 1 2\n1 0 3\n2 3 0\n")

    with pytest.raises(DatasetValidationError):
        load_distance_matrix(bad_matrix_path)


def test_optimal_tour_has_17_elements():
    tour = load_optimal_tour(OPTIMAL_TOUR_PATH)
    assert len(tour.one_indexed) == 17
    assert len(tour.zero_indexed) == 17


def test_optimal_tour_elements_are_distinct():
    tour = load_optimal_tour(OPTIMAL_TOUR_PATH)
    assert len(set(tour.one_indexed)) == 17


def test_optimal_tour_elements_between_1_and_17():
    tour = load_optimal_tour(OPTIMAL_TOUR_PATH)
    assert all(1 <= node <= 17 for node in tour.one_indexed)


def test_optimal_tour_zero_indexed_matches_one_indexed():
    tour = load_optimal_tour(OPTIMAL_TOUR_PATH)
    assert tour.zero_indexed == [node - 1 for node in tour.one_indexed]
