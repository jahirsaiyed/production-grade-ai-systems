import numpy as np


class DataValidationError(ValueError):
    """Raised when training data fails a validation gate."""


def validate_training_data(X: np.ndarray, y: np.ndarray) -> None:
    if X.shape[0] != y.shape[0]:
        raise DataValidationError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]} rows"
        )
    if np.isnan(X).any():
        raise DataValidationError("X contains NaN values")
    if np.isinf(X).any():
        raise DataValidationError("X contains infinite values")
    unique_labels = set(np.unique(y).tolist())
    if not unique_labels.issubset({0, 1}):
        raise DataValidationError(
            f"y contains labels outside {{0, 1}}: {unique_labels}"
        )
