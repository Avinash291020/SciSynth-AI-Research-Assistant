"""
Machine Learning Theory and Mathematical Foundations.

This module demonstrates:
- Linear algebra operations and concepts
- Loss functions and their derivatives
- Backpropagation implementation
- Optimization algorithms
- Mathematical foundations of ML
- Custom neural network implementations from scratch
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import math
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, log_loss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import sympy as sp
from sympy import symbols, diff, Matrix, simplify

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class Vector:
    """Custom vector class for linear algebra operations."""
    data: np.ndarray
    
    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
    
    def __add__(self, other):
        return Vector(self.data + other.data)
    
    def __sub__(self, other):
        return Vector(self.data - other.data)
    
    def __mul__(self, scalar):
        return Vector(self.data * scalar)
    
    def dot(self, other):
        return np.dot(self.data, other.data)
    
    def norm(self):
        return np.linalg.norm(self.data)
    
    def normalize(self):
        return Vector(self.data / self.norm())


class LinearAlgebra:
    """Linear algebra operations and concepts."""
    
    @staticmethod
    def matrix_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiplication with validation."""
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Matrix dimensions incompatible: {A.shape} and {B.shape}")
        return np.dot(A, B)
    
    @staticmethod
    def eigenvalue_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalue decomposition A = QΛQ^T."""
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        return eigenvalues, eigenvectors
    
    @staticmethod
    def singular_value_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Singular Value Decomposition A = UΣV^T."""
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        return U, S, Vt
    
    @staticmethod
    def matrix_rank(A: np.ndarray) -> int:
        """Calculate matrix rank."""
        return np.linalg.matrix_rank(A)
    
    @staticmethod
    def condition_number(A: np.ndarray) -> float:
        """Calculate condition number of matrix."""
        return np.linalg.cond(A)
    
    @staticmethod
    def projection_matrix(X: np.ndarray) -> np.ndarray:
        """Calculate projection matrix P = X(X^T X)^(-1) X^T."""
        XtX_inv = np.linalg.inv(np.dot(X.T, X))
        return np.dot(np.dot(X, XtX_inv), X.T)
    
    @staticmethod
    def gram_schmidt(vectors: List[np.ndarray]) -> List[np.ndarray]:
        """Gram-Schmidt orthogonalization."""
        ortho_vectors = []
        for v in vectors:
            v = v.copy()
            for u in ortho_vectors:
                v = v - np.dot(v, u) * u
            norm = np.linalg.norm(v)
            if norm > 1e-10:  # Avoid division by zero
                v = v / norm
                ortho_vectors.append(v)
        return ortho_vectors


class LossFunctions:
    """Implementation of common loss functions with derivatives."""
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error loss."""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Derivative of MSE with respect to predictions."""
        return 2 * (y_pred - y_true) / len(y_true)
    
    @staticmethod
    def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
        """Cross-entropy loss for classification."""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / len(y_true)
    
    @staticmethod
    def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
        """Derivative of cross-entropy with respect to predictions."""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / (y_pred * len(y_true))
    
    @staticmethod
    def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray, margin: float = 1.0) -> float:
        """Hinge loss for SVM."""
        return np.mean(np.maximum(0, margin - y_true * y_pred))
    
    @staticmethod
    def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
        """Huber loss for robust regression."""
        error = y_true - y_pred
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return np.mean(0.5 * quadratic**2 + delta * linear)
    
    @staticmethod
    def focal_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 1.0, gamma: float = 2.0) -> float:
        """Focal loss for handling class imbalance."""
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = (1 - pt) ** gamma
        return -np.mean(alpha * focal_weight * np.log(pt))


class ActivationFunctions:
    """Common activation functions with derivatives."""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function."""
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function."""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function."""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @staticmethod
    def softmax_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of softmax function."""
        s = ActivationFunctions.softmax(x)
        return s * (1 - s)


class NeuralNetworkFromScratch:
    """Neural network implementation from scratch with backpropagation."""
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """Initialize weights and biases using He initialization."""
        for i in range(len(self.layer_sizes) - 1):
            # He initialization for ReLU
            w = np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i]) * np.sqrt(2.0 / self.layer_sizes[i])
            b = np.zeros((self.layer_sizes[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward_propagation(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Forward propagation through the network."""
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            z_values.append(z)
            
            if i == len(self.weights) - 1:
                # Output layer - softmax for classification
                activation = ActivationFunctions.softmax(z)
            else:
                # Hidden layers - ReLU
                activation = ActivationFunctions.relu(z)
            
            activations.append(activation)
        
        return activations, z_values
    
    def backward_propagation(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], z_values: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward propagation to compute gradients."""
        m = X.shape[1]
        delta = activations[-1] - y  # Output layer error
        
        weight_gradients = []
        bias_gradients = []
        
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW = np.dot(delta, activations[i].T) / m
            db = np.sum(delta, axis=1, keepdims=True) / m
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            if i > 0:
                # Compute error for previous layer
                delta = np.dot(self.weights[i].T, delta) * ActivationFunctions.relu_derivative(z_values[i - 1])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients: List[np.ndarray], bias_gradients: List[np.ndarray]):
        """Update parameters using gradient descent."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int = 32) -> List[float]:
        """Train the neural network."""
        losses = []
        
        for epoch in range(epochs):
            # Mini-batch training
            for i in range(0, X.shape[1], batch_size):
                X_batch = X[:, i:i + batch_size]
                y_batch = y[:, i:i + batch_size]
                
                # Forward pass
                activations, z_values = self.forward_propagation(X_batch)
                
                # Backward pass
                weight_gradients, bias_gradients = self.backward_propagation(X_batch, y_batch, activations, z_values)
                
                # Update parameters
                self.update_parameters(weight_gradients, bias_gradients)
            
            # Compute loss
            activations, _ = self.forward_propagation(X)
            loss = LossFunctions.cross_entropy(y, activations[-1])
            losses.append(loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        activations, _ = self.forward_propagation(X)
        return activations[-1]


class OptimizationAlgorithms:
    """Implementation of various optimization algorithms."""
    
    @staticmethod
    def gradient_descent(f: Callable, grad_f: Callable, x0: np.ndarray, learning_rate: float = 0.01, max_iter: int = 1000) -> Tuple[np.ndarray, List[float]]:
        """Basic gradient descent."""
        x = x0.copy()
        history = []
        
        for i in range(max_iter):
            gradient = grad_f(x)
            x = x - learning_rate * gradient
            history.append(f(x))
        
        return x, history
    
    @staticmethod
    def momentum_gradient_descent(f: Callable, grad_f: Callable, x0: np.ndarray, learning_rate: float = 0.01, momentum: float = 0.9, max_iter: int = 1000) -> Tuple[np.ndarray, List[float]]:
        """Gradient descent with momentum."""
        x = x0.copy()
        velocity = np.zeros_like(x)
        history = []
        
        for i in range(max_iter):
            gradient = grad_f(x)
            velocity = momentum * velocity - learning_rate * gradient
            x = x + velocity
            history.append(f(x))
        
        return x, history
    
    @staticmethod
    def adam_optimizer(f: Callable, grad_f: Callable, x0: np.ndarray, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, max_iter: int = 1000) -> Tuple[np.ndarray, List[float]]:
        """Adam optimizer."""
        x = x0.copy()
        m = np.zeros_like(x)  # First moment
        v = np.zeros_like(x)  # Second moment
        history = []
        
        for t in range(1, max_iter + 1):
            gradient = grad_f(x)
            
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * gradient
            
            # Update biased second moment estimate
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1 ** t)
            
            # Compute bias-corrected second moment estimate
            v_hat = v / (1 - beta2 ** t)
            
            # Update parameters
            x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            history.append(f(x))
        
        return x, history


class MathematicalAnalysis:
    """Mathematical analysis and proofs."""
    
    @staticmethod
    def prove_linearity_of_expectation():
        """Demonstrate linearity of expectation."""
        # E[aX + bY] = aE[X] + bE[Y]
        X = np.random.normal(0, 1, 1000)
        Y = np.random.normal(0, 1, 1000)
        a, b = 2, 3
        
        left_side = np.mean(a * X + b * Y)
        right_side = a * np.mean(X) + b * np.mean(Y)
        
        print(f"E[aX + bY] = {left_side:.4f}")
        print(f"aE[X] + bE[Y] = {right_side:.4f}")
        print(f"Difference: {abs(left_side - right_side):.6f}")
        
        return abs(left_side - right_side) < 1e-10
    
    @staticmethod
    def prove_variance_formula():
        """Prove Var(X) = E[X²] - (E[X])²."""
        X = np.random.normal(0, 1, 1000)
        
        var_direct = np.var(X)
        var_formula = np.mean(X**2) - np.mean(X)**2
        
        print(f"Var(X) directly: {var_direct:.4f}")
        print(f"Var(X) using formula: {var_formula:.4f}")
        print(f"Difference: {abs(var_direct - var_formula):.6f}")
        
        return abs(var_direct - var_formula) < 1e-10
    
    @staticmethod
    def analyze_condition_number():
        """Analyze condition number and its effect on numerical stability."""
        # Well-conditioned matrix
        A_good = np.array([[1, 0.1], [0.1, 1]])
        cond_good = LinearAlgebra.condition_number(A_good)
        
        # Ill-conditioned matrix
        A_bad = np.array([[1, 0.999], [0.999, 1]])
        cond_bad = LinearAlgebra.condition_number(A_bad)
        
        print(f"Well-conditioned matrix condition number: {cond_good:.2f}")
        print(f"Ill-conditioned matrix condition number: {cond_bad:.2f}")
        
        return {"well_conditioned": cond_good, "ill_conditioned": cond_bad}


class VisualizationTools:
    """Tools for visualizing mathematical concepts."""
    
    @staticmethod
    def plot_loss_functions():
        """Plot different loss functions."""
        x = np.linspace(-2, 2, 100)
        y_true = np.array([1])
        
        plt.figure(figsize=(15, 10))
        
        # MSE
        plt.subplot(2, 3, 1)
        y_pred = x.reshape(-1, 1)
        mse_loss = [LossFunctions.mean_squared_error(y_true, np.array([yp])) for yp in y_pred]
        plt.plot(x, mse_loss)
        plt.title('Mean Squared Error')
        plt.xlabel('Prediction')
        plt.ylabel('Loss')
        
        # Cross-entropy
        plt.subplot(2, 3, 2)
        y_pred_sigmoid = ActivationFunctions.sigmoid(x).reshape(-1, 1)
        ce_loss = [LossFunctions.cross_entropy(y_true, np.array([yp])) for yp in y_pred_sigmoid]
        plt.plot(x, ce_loss)
        plt.title('Cross-Entropy Loss')
        plt.xlabel('Logit')
        plt.ylabel('Loss')
        
        # Hinge loss
        plt.subplot(2, 3, 3)
        hinge_loss = [LossFunctions.hinge_loss(y_true, np.array([yp])) for yp in y_pred]
        plt.plot(x, hinge_loss)
        plt.title('Hinge Loss')
        plt.xlabel('Prediction')
        plt.ylabel('Loss')
        
        # Activation functions
        plt.subplot(2, 3, 4)
        plt.plot(x, ActivationFunctions.sigmoid(x), label='Sigmoid')
        plt.plot(x, ActivationFunctions.relu(x), label='ReLU')
        plt.plot(x, ActivationFunctions.tanh(x), label='Tanh')
        plt.title('Activation Functions')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        
        # Gradients
        plt.subplot(2, 3, 5)
        plt.plot(x, ActivationFunctions.sigmoid_derivative(x), label='Sigmoid')
        plt.plot(x, ActivationFunctions.relu_derivative(x), label='ReLU')
        plt.plot(x, ActivationFunctions.tanh_derivative(x), label='Tanh')
        plt.title('Activation Function Derivatives')
        plt.xlabel('x')
        plt.ylabel("f'(x)")
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_optimization_comparison():
        """Compare different optimization algorithms."""
        # Define a simple function to optimize
        def f(x):
            return x[0]**2 + x[1]**2
        
        def grad_f(x):
            return np.array([2*x[0], 2*x[1]])
        
        x0 = np.array([2.0, 2.0])
        
        # Run different optimizers
        x_gd, hist_gd = OptimizationAlgorithms.gradient_descent(f, grad_f, x0, learning_rate=0.1)
        x_momentum, hist_momentum = OptimizationAlgorithms.momentum_gradient_descent(f, grad_f, x0, learning_rate=0.1)
        x_adam, hist_adam = OptimizationAlgorithms.adam_optimizer(f, grad_f, x0, learning_rate=0.1)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(hist_gd, label='Gradient Descent')
        plt.plot(hist_momentum, label='Momentum')
        plt.plot(hist_adam, label='Adam')
        plt.title('Optimization Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.legend()
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2
        
        plt.contour(X, Y, Z, levels=20)
        plt.plot([x0[0]], [x0[1]], 'ro', label='Start')
        plt.plot([x_gd[0]], [x_gd[1]], 'go', label='GD')
        plt.plot([x_momentum[0]], [x_momentum[1]], 'bo', label='Momentum')
        plt.plot([x_adam[0]], [x_adam[1]], 'mo', label='Adam')
        plt.title('Optimization Path')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Demonstrate linear algebra operations
    print("=== Linear Algebra Operations ===")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print("Matrix A:")
    print(A)
    print("Matrix B:")
    print(B)
    print("A * B:")
    print(LinearAlgebra.matrix_multiplication(A, B))
    
    # Demonstrate loss functions
    print("\n=== Loss Functions ===")
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    
    print(f"MSE: {LossFunctions.mean_squared_error(y_true, y_pred):.4f}")
    print(f"Cross-entropy: {LossFunctions.cross_entropy(y_true, y_pred):.4f}")
    
    # Demonstrate neural network from scratch
    print("\n=== Neural Network from Scratch ===")
    X = np.random.randn(2, 100)
    y = np.random.randint(0, 2, (1, 100))
    
    nn_scratch = NeuralNetworkFromScratch([2, 4, 2, 1], learning_rate=0.01)
    losses = nn_scratch.train(X, y, epochs=500, batch_size=32)
    
    # Demonstrate optimization algorithms
    print("\n=== Optimization Algorithms ===")
    def objective(x):
        return x[0]**2 + x[1]**2
    
    def gradient(x):
        return np.array([2*x[0], 2*x[1]])
    
    x0 = np.array([2.0, 2.0])
    x_opt, history = OptimizationAlgorithms.gradient_descent(objective, gradient, x0)
    print(f"Optimal point: {x_opt}")
    print(f"Final value: {objective(x_opt):.6f}")
    
    # Demonstrate mathematical proofs
    print("\n=== Mathematical Proofs ===")
    print("Linearity of expectation:", MathematicalAnalysis.prove_linearity_of_expectation())
    print("Variance formula:", MathematicalAnalysis.prove_variance_formula())
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    VisualizationTools.plot_loss_functions()
    VisualizationTools.plot_optimization_comparison()
    
    print("\nML Theory demonstration completed!") 