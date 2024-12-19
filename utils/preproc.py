import mlx.core as mx
import numpy as np


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
        # Convert to MLX array if needed
        if not isinstance(X, mx.array):
            X = mx.array(X)
        
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
        
        # Convert to MLX array if needed
        if not isinstance(X, mx.array):
            X = mx.array(X)
        
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
        
        # Convert to MLX array if needed
        if not isinstance(X_scaled, mx.array):
            X_scaled = mx.array(X_scaled)
        
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
    