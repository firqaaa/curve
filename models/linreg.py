import mlx.core as mx
import wandb


class LinearRegression:
    def __init__(self, learning_rate=0.01, num_epochs=1000, batch_size=None):
        """
        Initialize Linear Regression model

        Args:
            learning_rate (float): Learning rate for gradient descent
            num_epochs (int): Number of training iterations
            batch_size (int): Size of batches for mini-batch gradient descent
        """
        self.config = {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "optimizer": "gradient_descent",
            "loss": "mse",
            "arch": "linear_regression"
        }
        self.weights = None
        self.bias = None
        self.history = {'loss': [], 'val_loss': [], 'r2': [], 'val_r2': []}

    def _initialize_parameters(self, n_features):
        """
        Initialize weights and bias

        Args:
            n_features (int): Number of input features
        """
        # Initialize weights and bias with random values
        self.weights = mx.random.normal((n_features, 1)) * 0.01
        self.bias = mx.zeros((1, ))

        wandb.config.update({
            "n_features": n_features,
            "n_parameters": n_features + 1 # weight + bias
        })
    
    def _mean_squared_error(self, y_pred, y_true):
        """
        Calculate mean squared error loss

        Args:
            y_pred (mx.array): Predicted values
            y_true (mx.array): True values
        
        Returns:
            float: Mean squared error
        """
        return mx.mean((y_pred - y_true) ** 2)
    
    def _forward(self, X):
        """
        Forward pass to compute predictions

        Args:
            X (mx.array): Input features
        
        Returns:
            mx.array: Predicted values
        """
        return mx.matmul(X, self.weights) + self.bias
    
    def _calculate_metrics(self, X, y, prefix=""):
        """
        Calculate various metrics

        Args:
            X (mx.array): Features
            y (mx.array): True values
            prefix (str): Prefix for metric names (e.g., 'val_' for validation)
        
        Returns:
            dict: Dictionary of metrics
        """
        y_pred = self.predict(X)

        # Calculate MSE
        mse = float(self._mean_squared_error(y_pred, y))

        # Calculate R-squared
        ss_total = mx.sum((y - mx.mean(y)) ** 2)
        ss_residual = mx.sum((y - y_pred) ** 2)
        r2 = float(1 - (ss_residual / ss_total))

        # Calculate Mean Absolute Error
        mae = float(mx.mean(mx.abs(y_pred - y)))

        # Calculate Root Mean Squared Error
        rmse = float(mx.sqrt(mse))

        metrics = {
            f"{prefix}mse": mse,
            f"{prefix}r2": r2,
            f"{prefix}mae": mae,
            f"{prefix}rmse": rmse
        }

        return metrics        

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train the linear regression model

        Args:
            X_train (mx.array): Training features
            y_train (mx.array): Target values
            X_val (mx.array): Validation features
            y_val (mx.array): Validation target values
            verbose (bool): Whether to print training progress
        """
        # Initialize wandb run if not already initialized
        if wandb.run is None:
            wandb.init(project="linear-regression-mlx", config=self.config)
        
        # Reshape y if needed
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        
        # Get number of features and initialize parameters
        n_features = X_train.shape[1]
        self._initialize_parameters(n_features)

        # Handle validation data
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            X_val = mx.array(X_val) if not isinstance(X_val, mx.array) else X_val
            y_val = mx.array(y_val) if not isinstance(y_val, mx.array) else y_val
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)
        
        # Training Loop
        for epoch in range(self.config["num_epochs"]):
            # Forward pass
            y_pred = self._forward(X_train)

            # Compute gradients
            dw = (2/X_train.shape[0]) * mx.matmul(X_train.T, (y_pred - y_train))
            db = (2/X_train.shape[0]) * mx.sum(y_pred - y_train)

            # Update parameters
            self.weights -= self.config["learning_rate"] * dw
            self.bias -= self.config["learning_rate"] * db

            # Calculate and log metrics
            metrics = self._calculate_metrics(X_train, y_train)

            if has_validation:
                val_metrics = self._calculate_metrics(X_val, y_val, prefix='val_')
                metrics.update(val_metrics)
            
            # Add epoch number to metrics
            metrics['epoch'] = epoch + 1

            # Log weight norms
            metrics['weight_norm'] = float(mx.sum(self.weights ** 2))

            # Log to wandb
            wandb.log(metrics)

            # Store in history
            for key, value in metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{self.config["num_epochs"]}], "
                      f"Loss: {metrics["mse"]:.4f}, "
                      f"R2: {metrics["r2"]:.4f}"
                      + (f', Val Loss: {metrics["val_mse"]:.4f}, '
                         f'Val R2: {metrics["val_r2"]:.4f}'
                         if has_validation else ''))
    
    def predict(self, X):
        """Make predictions"""
        if not isinstance(X, mx.array):
            X = mx.array(X)
        return self._forward(X)
    
                