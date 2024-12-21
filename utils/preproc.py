import mlx.core as mx
import numpy as np
import pandas as pd


class StandardScaler:
    def __init__(self):
        """
        Initialize scaler with empty mean and std attributes
        """
        self.mean = None
        self.std_ = None
        self.n_features = None

    def fit(self, X):
        """
        Calculates mean and standard deviation of features

        Args:
            X (mx.array): Input features of shape (n_samples, n_features)
        """

        # Handle different input types
        if isinstance(X, pd.DataFrame):
            X_values = X.values.astype(np.float32)
        elif isinstance(X, np.ndarray):
            X_values = X.astype(np.float32)
        elif isinstance(X, mx.array):
            X_values = X
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")
        
        # Convert to MLX array
        try:
            X = mx.array(X_values)
        except Exception as e:
            print(f"Error converting to MLX array: {e}")
            print(f"Input shape: {X_values.shape}, dtype: {X_values.dtype}")
            raise

        # Store number of features
        self.n_features = X.shape[1]

        # Calculate mean along each feature
        self.mean_ = mx.mean(X, axis=0)

        # Calculate standard deviation along each feature
        self.std_ = mx.std(X, axis=0)

        # Handle zero standard deviation
        self.std_ = mx.where(self.std_ == 0, 1.0, self.std_)

        return self
    
    def transform(self, X):
        """
        Standardize features by removing mean and scaling to unit variance

        Args:
            X (mx.array): Input features of shape (n_samples, n_features)
        
        Returns:
            mx.array: Standardize features
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        # Handle different input types
        if isinstance(X, pd.DataFrame):
            X_values = X.values.astype(np.float32)
        elif isinstance(X, np.ndarray):
            X_values = X.astype(np.float32)
        elif isinstance(X, mx.array):
            X_values = X
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

        # Convert to MLX array
        try:
            X = mx.array(X_values)
        except Exception as e:
            print(f"Error converting to MLX array: {e}")
            print(f"Input shape: {X_values.shape}, dtype: {X_values.dtype}")
            raise
        
        # Check feature dimension
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features}, got {X.shape[1]}")
        
        # Standardize features
        X_scaled = (X - self.mean_) / self.std_

        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit scaler and standardize features in one go

        Args:
            X (mx.array): Input features of shape (n_samples, n_features)

        Returns:
            mx.array: Standardize features
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Scale back standardize features to original scale

        Args:
            X_scaled (mx.array): Standardize features
        
        Returns:
            mx.array: Features in original scale
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        # Handle different input types
        if isinstance(X_scaled, pd.DataFrame):
            X_values = X_scaled.values.astype(np.float32)
        elif isinstance(X_scaled, np.ndarray):
            X_values = X_scaled.astype(np.float32)
        elif isinstance(X_scaled, mx.array):
            X_values = X_scaled
        else:
            raise ValueError(f"Unsupported input type: {type(X_scaled)}")

        # Convert to MLX array
        try:
            X_scaled = mx.array(X_values)
        except Exception as e:
            print(f"Error converting to MLX array: {e}")
            print(f"Input shape: {X_values.shape}, dtype: {X_values.dtype}")
            raise
        
        # Inverse transform
        X = (X_scaled * self.std_) + self.mean_

        return X
    

class LabelEncoder:
    def __init__(self):
        """Initialize LabelEncoder with empty classes"""
        self.classes_ = None
        self.class_to_index = None
        self.index_to_class = None

    def fit(self, y):
        """
        Fit label encoder by finding unique classes

        Args:
            y (array-like): Target values to encode
        
        Returns:
            self: Returns the instance itself
        """
        # Convert to list if input is MLX array
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(y, mx.array):
            y = y.tolist()
        
        # Convert all elements to strings for consistent sorting
        str_labels = [str(label) for label in y]

        # Get unique classes and sort them
        unique_classes = sorted(set(str_labels))
        self.classes_ = unique_classes

        # Store original type for each class by checking the first occurence
        self.classes_ = []
        for class_str in unique_classes:
            # Find first occurence of this value in original data
            original_value = next(val for val in y if str(val) == class_str)
            self.classes_.append(original_value)

        # Create mapping dictionaries
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.index_to_class = {idx: cls for idx, cls in enumerate(self.classes_)}

        return self

    def transform(self, y):
        """
        Transform labels to normalized encoding

        Args:
            y (array-like): Target values to encode
        
        Returns:
            mx.array: Encoded labels
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder has not been fitted yet.")
        
        if isinstance(y, pd.Series):
            y = y.values

        # Convert to list if input is MLX array
        if isinstance(y, mx.array):
            y = y.tolist()
        
        # Handle single input
        if not isinstance(y, (list, np.ndarray)):
            y = [y]
        
        try:
            # Transform labels to indices
            encoded = [self.class_to_index[label] for label in y]
            return mx.array(encoded)
        except KeyError as e:
            raise ValueError(f"Unkwon label: {str(e)}")
        
    def fit_transform(self, y):
        """
        Fit label encoder and return encoded labels

        Args:
            y (array-like): Target values to encode
        
        Returns:
            mx.array: Encoded labels
        """
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        """
        Convert the encoded labels back to original encoding

        Args:
            y (array-like): Encoded values to decode
        
        Returns:
            list: Original labels
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder has not been fitted yet.")
        
        if isinstance(y, pd.Series):
            y = y.values
        # Convert to list if input is MLX array
        if isinstance(y, mx.array):
            y = y.tolist()
        
        # Handle single output
        if not isinstance(y, (list, np.ndarray)):
            y = [y]
        
        try:
            # Transform indices back to original labels
            decoded = [self.index_to_class[int(idx)] for idx in y]
            return decoded
        except KeyError as e:
            raise ValueError(f"Unkwon encoded value: {str(e)}")

    def get_classes(self):
        """
        Get the classes seen during fitting

        Returns:
            list: list of classes
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder has not been fitted yet.")
        return self.classes_

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        """
        Initialize MinMaxScaler

        Args:
            feature_range (tuple): Desired range of transformed data, default (0, 1)
        """
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.n_features = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, X):
        """
        Compute the minimum and maximum values to be used for scaling

        Args:
            X: Input data (DataFrame, ndarray, or mx.array)
        
        Returns:
            self. Return the scaler object
        """
        # Handle different input types
        if isinstance(X, pd.DataFrame):
            X_values = X.values.astype(np.float32)
        elif isinstance(X, np.ndarray):
            X_values = X.astype(np.float32)
        elif isinstance(X, mx.array):
            X_values = X
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")
        
        # Convert to MLX array
        try:
            X = mx.array(X_values)
        except:
            print(f"Error converting to MLX array: {e}")
            print(f"Input shape: {X_values.shape}, dtype: {X_values.dtype}")
            raise
    
        # Store number of features
        self.n_features = X.shape[1] if len(X.shape) > 1 else 1

        # Compute data min and max
        self.data_min_ = mx.min(X, axis=0)
        self.data_max_ = mx.max(X, axis=0)

        # Compute range
        self.data_range_ = self.data_max_ - self.data_min_

        # Handle constant features
        self.data_range_ = mx.where(self.data_range_ == 0, 1.0, self.data_range_)

        # Compute scale and min for transformation
        feature_range_min, feature_range_max = self.feature_range
        self.scale_ = (feature_range_max - feature_range_min) / self.data_range_
        self.min_ = feature_range_min - self.data_min_ * self.scale_

        return self
    
    def transform(self, X):
        """
        Scale features according to feature_range

        Args:
            X: Input data (DataFrame, ndarray, or mx.array)
        
        Returns:
            mx.array: Scaled features
        """
        if self.scale_ is None:
            raise ValueError("MinMaxScaler is not fitted yet. Call it first.")

        # Handle different input types
        if isinstance(X, pd.DataFrame):
            X_values = X.values.astype(np.float32)
        elif isinstance(X, np.ndarray):
            X_values = X.astype(np.float32)
        elif isinstance(X, mx.array):
            X_values = X
        else:
            raise ValueError(f"Unsupported input types: {type(X)}")
        
        # Convert to MLX array
        try:
            X = mx.array(X_values)
        except Exception as e:
            print(f"Error converting to MLX array: {e}")
            print(f"Input shape: {X_values.shape}, dtype: {X_values.dtype}")
            raise
        
        # Check feature dimension
        if len(X.shape) > 1 and X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
        
        # Transform data
        X_scaled = X * self.scale_ + self.min_

        return X_scaled

    def fit_transform(self, X):
        """
        Fit to data, then transform it

        Args:
            X: Input data (DataFrame, ndarray, or mx.array)
        
        Returns:
            mx.array: Scaled features
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Undo the scaling of X according to feature_range

        Args:
            X_scaled: Scaled input data (DataFrame, ndarray, or mx.array)
        
        Returns:
            mx.array: Original features
        """
        if self.scale_ is None:
            raise ValueError("MinMaxScaler is not fitted yet. Call it first.")
        
        # Handle different input types
        if isinstance(X_scaled, pd.DataFrame):
            X_values = X_scaled.values.astype(np.float32)
        elif isinstance(X_scaled, np.ndarray):
            X_values = X_scaled.astype(np.float32)
        elif isinstance(X_scaled, mx.array):
            X_values = X_scaled
        else:
            raise ValueError(f"Unsupported input type: {type(X_scaled)}")
        
        # Convert to MLX array
        try:
            X_scaled = mx.array(X_values)
        except Exception as e:
            print(f"Error converting to MLX array: {e}")
            print(f"Input shape: {X_values.shape}, dtype: {X_values.dtype}")
            raise

        # Inverse transform
        X = (X_scaled - self.min_) / self.scale_

        return X