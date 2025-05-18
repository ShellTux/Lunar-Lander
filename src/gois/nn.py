import numpy as np

class NeuralNetwork:
    shape: list[int]
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    activations: list[np.ndarray]

    def __init__(self, *sizes: int):
        assert len(sizes) >= 3

        self.shape = list(sizes)
        self.weights = [
            np.zeros((layer_size, next_layer_size), dtype=np.float32)
            for layer_size, next_layer_size in zip(self.shape[:-1], self.shape[1:])
        ]
        self.biases = [
            np.zeros((1, layer_size)).astype(np.float32)
            for layer_size in self.shape[1:]
        ]

        assert len(self.weights) == len(self.biases)

    def __eq__(self, other):
        if not isinstance(other, NeuralNetwork):
            return False

        return all([
            self.shape == other.shape,
            np.all(self.weights == other.weights),
            np.all(self.biases == other.biases),
        ])

    def __str__(self) -> str:
        """String representation of the network architecture."""
        inputs = self.shape[0]
        outputs = self.shape[-1]
        hidden = self.shape[1:-1]
        weights = '\n\n'.join(np.array2string(weight, precision=2) for weight in self.weights)
        return f'''
Neural Network Architecture: {self.shape}
    - inputs :       {inputs}
    - hidden layers: {len(hidden)} {hidden}
    - outputs:       {outputs}

weights: [
{weights}
]
'''.strip()

    @staticmethod
    def random(*sizes: int):
        nn = NeuralNetwork(*sizes)

        nn.weights = [
            np.random.uniform(-1, 1, size=(layer_size, next_layer_size)).astype(np.float32)
            for layer_size, next_layer_size in zip(nn.shape[:-1], nn.shape[1:])
        ]
        nn.biases = [
            np.zeros((1, layer_size)).astype(np.float32)
            for layer_size in nn.shape[1:]
        ]

        return nn

    def set_parameters(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        """Set weights and biases manually."""
        assert len(weights) == len(self.weights), "Number of weight matrices must match."
        assert len(biases) == len(self.biases), "Number of bias vectors must match."

        for weight, bias in zip(weights, biases):
            assert weight.shape[1] == bias.shape[1], "Bias shape must match weight's column size."

        self.weights = weights
        self.biases = biases

    def copy(self):
        nn = NeuralNetwork(1,1,1)
        nn.shape = self.shape
        nn.weights = self.weights.copy()
        nn.biases = self.biases.copy()

        return nn

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        activation = lambda x: self._sigmoid(x)

        self.activations = [X]

        for weight, bias in zip(self.weights, self.biases):
            # print(f'{X.shape=} * {weight.shape=} + {bias.shape=}', file=stderr)
            X = X @ weight + bias
            X = activation(X)

            self.activations.append(X)

        # assert X.shape[0] == 1

        return X.flatten()

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of the sigmoid activation function."""
        return z * (1 - z)

    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01) -> None:
        """Backward pass for training the network using gradient descent."""
        inverse_activation = lambda x: self._sigmoid_derivative(x)

        m = X.shape[0]  # Number of examples
        output = self.activations[-1]

        # Calculate output layer error
        output_error = output - y
        output_delta = output_error * self._sigmoid_derivative(output)

        # Update weights and biases for output layer
        self.weights[-1] -= (self.activations[-2].T @ output_delta) * (learning_rate / m)
        self.biases[-1] -= np.sum(output_delta, axis=0, keepdims=True) * (learning_rate / m)

        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            hidden_error = output_delta @ self.weights[i + 1].T
            hidden_delta = hidden_error * self._sigmoid_derivative(self.activations[i + 1])

            # Update weights and biases for hidden layers
            self.weights[i] -= (self.activations[i].T @ hidden_delta) * (learning_rate / m)
            self.biases[i] -= np.sum(hidden_delta, axis=0, keepdims=True) * (learning_rate / m)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, learning_rate: float = 0.01) -> None:
        """Train the network on the given data."""
        X = X.copy()

        for _ in range(epochs):
            np.random.shuffle(X)
            for inputs in X:
                self.forward(inputs)  # Forward pass
                self.backward(inputs, y, learning_rate)  # Backward pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the network."""
        return self.forward(X)
