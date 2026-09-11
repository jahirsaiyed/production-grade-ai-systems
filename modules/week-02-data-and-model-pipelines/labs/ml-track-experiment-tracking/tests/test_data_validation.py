import numpy as np
import pytest

from data_validation import DataValidationError, validate_training_data


def test_validate_training_data_passes_for_clean_data():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    validate_training_data(X, y)


def test_validate_training_data_rejects_mismatched_row_counts():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0])
    with pytest.raises(DataValidationError):
        validate_training_data(X, y)


def test_validate_training_data_rejects_nan():
    X = np.array([[1.0, np.nan], [3.0, 4.0]])
    y = np.array([0, 1])
    with pytest.raises(DataValidationError):
        validate_training_data(X, y)


def test_validate_training_data_rejects_infinite_values():
    X = np.array([[1.0, np.inf], [3.0, 4.0]])
    y = np.array([0, 1])
    with pytest.raises(DataValidationError):
        validate_training_data(X, y)


def test_validate_training_data_rejects_unexpected_labels():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 2])
    with pytest.raises(DataValidationError):
        validate_training_data(X, y)
