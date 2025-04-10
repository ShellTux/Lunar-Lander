import argparse
import gymnasium as gym
import numpy as np

parser = argparse.ArgumentParser(description='Lunar Lander')
parser.add_argument(
    '--render',
    choices=['none', 'human'],
    default='human',
    help="Set the render option to 'none' or 'human'."
)
parser.add_argument(
    '--episodes',
    type=int,
    default=1000,
    help="Set the number of episodes."
)
parser.add_argument(
    '--gravity',
    type=float,
    default=-10.0,
    help="Set the gravity."
)
parser.add_argument(
    '--turbulence-power',
    type=float,
    default=0.0,
    help="Set the turbulence power."
)
parser.add_argument(
    '--wind-power',
    type=float,
    default=15.0,
    help="Set the wind power."
)
parser.add_argument(
    '--wind',
    type=bool,
    default=False,
    help="Enable the wind."
)
parser.add_argument(
    '--evolve',
    type=bool,
    default=True,
    help="Enable Evolution."
)
parser.add_argument(
    '--generations',
    type=int,
    default=100,
    help="Set the number of generations."
)
parser.add_argument(
    '--crossover-probability',
    type=float,
    default=.7,
    help="Set the number of crossover probability."
)
parser.add_argument(
    '--mutation-probability',
    type=float,
    default=.1,
    help="Set the number of generations."
)

args = parser.parse_args()

class NeuralNetwork:
    shape: list[int]
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    activations: list[np.ndarray]

    def __init__(self, *sizes: int):
        assert len(sizes) >= 3

        self.shape = list(sizes)
        self.weights = [
            np.random.uniform(-1, 1, size=(layer_size, next_layer_size)).astype(np.float32)
            for layer_size, next_layer_size in zip(self.shape[:-1], self.shape[1:])
        ]
        self.biases = [
            np.zeros((1, layer_size)).astype(np.float32)
            for layer_size in self.shape[1:]
        ]

        assert len(self.weights) == len(self.biases)

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
#
#
# biases: [
# {'\n\n'.join(np.array2string(bias, precision=2) for bias in self.biases)}
# ]
#     '''

    def set_parameters(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        """Set weights and biases manually."""
        assert len(weights) == len(self.weights), "Number of weight matrices must match."
        assert len(biases) == len(self.biases), "Number of bias vectors must match."

        for weight, bias in zip(weights, biases):
            assert weight.shape[1] == bias.shape[1], "Bias shape must match weight's column size."

        self.weights = weights
        self.biases = biases

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

def single_random_agent_example():
    agent = NeuralNetwork(8, 3, 3, 2)
    print(agent)

    env = gym.make(
        "LunarLander-v3",
        render_mode=args.render,
        gravity=args.gravity,
        continuous=True,
        enable_wind=args.wind,
        wind_power=args.wind_power,
        turbulence_power=args.turbulence_power
    )

    observation, _ = env.reset()

    while True:
        action = agent.forward(observation)
        # action = np.random.uniform(-1, 1, size=(2,)).astype(np.float32).clip(-1, 1).round(decimals=2)
        print(f'action={np.array2string(action, precision=2)}', end='\r')

        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            break

    print()
    env.close()

if __name__ == "__main__":
    single_random_agent_example()
