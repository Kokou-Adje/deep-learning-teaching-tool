"""
Neural Network implementation with backpropagation
"""

import numpy as np


class NeuralNetwork:
    """
    Feedforward Neural Network with one hidden layer
    Architecture: input(2) -> hidden(n) -> output(1)
    """
    
    def __init__(self, input_size=2, hidden_size=4, output_size=1, random_seed=42):
        """
        Initialize network with random weights
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden neurons
            output_size: Number of output neurons
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Store activations for visualization
        self.hidden_activations = None
        self.output_activation = None
        self.loss_history = []
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        """Derivative of sigmoid function"""
        return a * (1 - a)
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: Input data (n_samples, input_size)
        
        Returns:
            Output predictions (n_samples, output_size)
        """
        # Hidden layer
        z1 = np.dot(X, self.W1) + self.b1
        self.hidden_activations = self.sigmoid(z1)
        
        # Output layer
        z2 = np.dot(self.hidden_activations, self.W2) + self.b2
        self.output_activation = self.sigmoid(z2)
        
        return self.output_activation
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss
        
        Args:
            y_true: True labels (n_samples,)
            y_pred: Predicted probabilities (n_samples, 1)
        
        Returns:
            Average loss across samples
        """
        y_true = y_true.reshape(-1, 1)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Prevent log(0)
        
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss
    
    def backward(self, X, y, learning_rate):
        """
        Backpropagation - compute gradients and update weights
        
        Args:
            X: Input data (n_samples, input_size)
            y: True labels (n_samples,)
            learning_rate: Learning rate for gradient descent
        
        Returns:
            Loss value
        """
        m = X.shape[0]  # Number of samples
        y = y.reshape(-1, 1)
        
        # Forward pass
        y_pred = self.forward(X)
        
        # Compute loss
        loss = self.compute_loss(y, y_pred)
        
        # Backward pass - Output layer
        dZ2 = y_pred - y  # Derivative of loss w.r.t. output
        dW2 = np.dot(self.hidden_activations.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Backward pass - Hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.hidden_activations)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
        return loss
    
    def train(self, X, y, learning_rate=0.5, epochs=1):
        """
        Train the network for multiple epochs
        
        Args:
            X: Training data (n_samples, input_size)
            y: Training labels (n_samples,)
            learning_rate: Learning rate
            epochs: Number of training epochs
        
        Returns:
            List of losses for each epoch
        """
        losses = []
        for epoch in range(epochs):
            loss = self.backward(X, y, learning_rate)
            losses.append(loss)
        
        return losses
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input data (n_samples, input_size)
        
        Returns:
            Binary predictions (n_samples,)
        """
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """
        Predict probabilities
        
        Args:
            X: Input data (n_samples, input_size)
        
        Returns:
            Probability predictions (n_samples,)
        """
        return self.forward(X).flatten()
    
    def get_decision_boundary(self, resolution=100):
        """
        Compute decision boundary for visualization
        
        Args:
            resolution: Grid resolution
        
        Returns:
            xx, yy: Coordinate meshgrids
            Z: Predictions for each point
        """
        x_min, x_max = 0, 1
        y_min, y_max = 0, 1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict_proba(X_grid)
        Z = Z.reshape(xx.shape)
        
        return xx, yy, Z
    
    def get_accuracy(self, X, y):
        """
        Compute accuracy
        
        Args:
            X: Input data
            y: True labels
        
        Returns:
            Accuracy (0 to 1)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)