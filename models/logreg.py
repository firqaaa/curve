import mlx.core as mx


class LogisticRegression:
    def __init__(self, input_dim, learning_rate=0.01, num_epochs=100):
        """
        Initialize Logistic Regression model

        Args:
            input_dim (int): Number of input features
            learning_rate (float): Learning rate for gradient descent
            num_epochs (int): Number of training epochs
        """
        self.weights = mx.zeros((input_dim, 1))
        self.bias = mx.zeros((1, ))
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def sigmoid(self, z):
        """
        Sigmoid activation function

        Args:
            z (mx.array): Input array
        
        Returns:
            mx.array: Sigmoid of input
        """
        return 1 / (1 + mx.exp(-z))
    
    def binary_cross_entropy_loss(self, y_pred, y_true):
        """
        Compute binary cross-entropy loss

        Args:
            y_pred (mx.array): Predicted probabilities
            y_true (mx.array): True labels

        Returns:
            floats: Loss value
        """
        epsilon = 1e-15
        y_pred = mx.clip(y_pred, epsilon, 1 - epsilon)
        loss = -(y_true * mx.log(y_pred) + (1 - y_true) * mx.log(1 - y_pred))
        return mx.mean(loss)
    
    def fit(self, X, y, X_val, y_val, wandb_run, logging=False):
        """
        Train logistic regression model

        Args:
            X (mx.array): Input features
            y (mx.array): Binary labels
        """
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(y_val.shape) == 1:
            y_val = y_val.reshape(-1, 1)

        for epoch in range(self.num_epochs):
            # Training
            z = mx.matmul(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            # Compute gradients
            dw = (1/X.shape[0]) * mx.matmul(X.T, (y_pred - y))
            db = (1/X.shape[0]) * mx.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.bias * db

            # Compute training metrics
            train_loss = self.binary_cross_entropy_loss(y_pred, y)
            train_accuracy = mx.mean((y_pred >= 0.5).astype(mx.float32) == y)

            # Compute validation metrics
            val_pred = self.predict(X_val)
            val_loss = self.binary_cross_entropy_loss(val_pred, y_val)
            val_accuracy = mx.mean((val_pred >= 0.5).astype(mx.float32) == y_val)

            if logging:
                wandb_run.log({
                    'epoch': epoch,
                    'train_loss': train_loss.item(),
                    'train_accuracy': train_accuracy.item(),
                    'val_loss': val_loss.item(),
                    'val_accuracy': val_accuracy.item()
                })  
                
                if epoch % 1 == 0:
                    print(f"Epoch {epoch}, Train Loss: {train_loss.item():.4f}, "
                          f"Train Acc: {train_accuracy.item():.4f}, "
                          f"Val Loss: {val_loss.item():.4f}, "
                          f"Val Acc: {val_accuracy.item():.4f}")
    
    def predict(self, X):
        z = mx.matmul(X, self.weights) + self.bias
        return self.sigmoid(z)