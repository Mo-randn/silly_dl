import abc
import numpy as np

class Layer(abc.ABC):
    def __init__(self):
        super().__init__()
        # Initialize any common state here if needed (often nothing).
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def backward(self, grad_output):
        pass

    @abc.abstractmethod
    def update_params(self, learning_rate):
        pass

    @abc.abstractmethod
    def parameters(self):
        pass

class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        import numpy as np
        self.weights = np.random.randn(input_dim, output_dim) * 0.1
        self.bias = np.zeros((1, output_dim))

        # Same shape as parameters for storing gradients
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias    = np.zeros_like(self.bias)

    def forward(self, x):
        # Save input 'x' if needed for backward
        self.x = x
        return x.dot(self.weights) + self.bias

    def backward(self, grad_output):
        # grad_output has shape (batch_size, output_dim)
        # self.x has shape (batch_size, input_dim)
        self.grad_weights = self.x.T.dot(grad_output) 
        self.grad_bias = grad_output.sum(axis=0, keepdims=True)     

        # Gradient wrt input to this layer
        grad_input = grad_output.dot(self.weights.T) 
        return grad_input

    def update_params(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.bias    -= learning_rate * self.grad_bias

    def parameters(self):
        return [self.weights, self.bias]


class ReLULayer(Layer):
    """
    Simple ReLU activation layer. No trainable parameters.
    """

    def __init__(self):
        super().__init__()
        # No weights or biases to initialize

    def forward(self, x):

        self.x = x  # store for backward
        return np.maximum(0, x)

    def backward(self, grad_output):
        # dL/dx = dL/dy * dy/dx
        # ReLU'(x) = 1 if x > 0, else 0
        relu_mask = (self.x > 0).astype(grad_output.dtype)
        grad_input = grad_output * relu_mask
        return grad_input

    def update_params(self, learning_rate):
        # No parameters to update
        pass

    def parameters(self):
        # No trainable parameters
        return []

    
class SoftmaxCrossEntropyLayer(Layer):
    """
    Combined Softmax + Cross-Entropy "layer"
    No trainable parameters.
    """

    def __init__(self):
        super().__init__()
        # No weights or biases
        # We'll store references during forward to use in backward
        self.logits = None
        self.labels = None
        self.probs = None
        self.batch_size = None

    def forward(self, x, labels):

        self.logits = x
        self.labels = labels
        self.batch_size = x.shape[0]

        # Numerically stable softmax
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        self.probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # Cross-entropy loss = -sum(y * log(probs)) / batch_size
        eps = 1e-9  # to avoid log(0)
        log_likelihood = -np.log(self.probs + eps)
        loss = np.mean(np.sum(labels * log_likelihood, axis=1))
        return loss

    def backward(self, grad_output=1.0):

        # dL/dx = (probs - labels) / batch_size  (chain rule: * grad_output)
        grad_input = (self.probs - self.labels) / self.batch_size
        return grad_input * grad_output  # Multiply by grad_output if needed

    def update_params(self, learning_rate):
        # No parameters to update
        pass

    def parameters(self):
        # No trainable parameters
        return []