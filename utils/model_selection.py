import mlx.core as mx
import numpy as np


def train_test_split(X, y, test_size=0.2, random_seed=None):
    """
    Split arrays into random train and test subsets

    Args:
        X (mx.array): Features array
        y (mx.array): Target array
        test_size (float): Proportion of dataset to include in test split
        random_seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Convert inputs to MLX arrays if needed
    if not isinstance(X, mx.array):
        X = mx.array(X)
    if not isinstance(y, mx.array):
        y = mx.array(y)
    
    X = np.array(X)
    y = np.array(y)

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Get number of samples
    n_samples = X.shape[0]

    # Calculate number of test samples
    n_test = int(n_samples * test_size)

    # Generate random indices
    indices = np.random.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Split the data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return (mx.array(X_train), mx.array(X_test), mx.array(y_train), mx.array(y_test))