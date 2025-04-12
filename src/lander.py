from args import args
from nn import NeuralNetwork
from pathlib import Path
import gymnasium as gym
import numpy as np

class Lander:
    nn: NeuralNetwork
    fitness: float

    def __init__(self, *shape: int) -> None:
        # Initialize the neural network for the lander agent
        self.nn = NeuralNetwork.random(*shape) if len(shape) >= 3 else NeuralNetwork.random(8, 6, 6, 2)
        self.fitness = 0  # Track the fitness score of the lander

    def __eq__(self, other):
        if not isinstance(other, Lander):
            return False

        return all([
            self.nn == other.nn,
            self.fitness == other.fitness,
        ])

    def serialize(
        self,
        filename: str | Path,
        *,
        append: bool = False,
    ) -> None:
        """Serialize the Lander to a file."""
        mode = 'a' if append else 'w'
        with open(filename, mode) as f:
            # Serialize shape
            shape_str = ' '.join(map(str, self.nn.shape))

            # Serialize weights and biases
            weights_str = ' | '.join(' '.join(map(str, weight.flatten())) for weight in self.nn.weights)
            biases_str = ' | '.join(' '.join(map(str, bias.flatten())) for bias in self.nn.biases)

            # Write all parts on one line
            f.write(f"{self.fitness} | {shape_str} | {weights_str} | {biases_str}\n")

    @staticmethod
    def deserialize(filename: str | Path):
        """Deserialize the Lander from a file."""
        lander = Lander()

        with open(filename, 'r') as f:
            lines = f.readlines()
            line = lines[-1].strip()

            parts = line.split(' | ')

            # Deserialize shape
            lander.fitness = int(parts[0])
            shape = [int(size) for size in parts[1].split()]
            network = NeuralNetwork(*shape)

            # Deserialize weights
            weight_data = parts[2:]
            index = 0
            for i, _ in enumerate(network.weights):
                # Calculate number of elements in weight matrix
                network.weights[i] = np.array([
                    float(weight) for weight in weight_data[index].split()
                ]).reshape(network.shape[i], network.shape[i + 1])
                index += 1

            # Deserialize biases
            bias_data = weight_data[index:]
            for i, _ in enumerate(network.biases):
                network.biases[i] = np.array([
                    float(bias) for bias in bias_data[i].split()
                ]).reshape(1, network.shape[i + 1])

            lander.nn = network

        return lander

    def set_fitness(self, fitness: float) -> None:
        """Set the fitness score of the lander."""
        self.fitness = fitness

    def mutate(self, mutation_rate: float) -> None:
        for i in range(len(self.nn.weights)):
            if np.random.rand() > mutation_rate:
                continue

            mutation_amount: np.ndarray = np.random.normal(0, .1, size=self.nn.weights[i].shape)
            self.nn.weights[i] = (self.nn.weights[i] + mutation_amount).clip(-1, 1)

    def run(
        self,
        *,
        env: gym.Env,
        debug: bool = False,
        seed: int | None = None,
    ):
        observation, _ = env.reset(seed=seed)

        total_reward: float = 0
        time_survived: int = 0
        penalties: float = 0

        landed: bool = True

        while True:
            action = self.nn.forward(observation)
            # action = np.random.uniform(-1, 1, size=(2,)).astype(np.float32).clip(-1, 1).round(decimals=2)

            observation, reward, terminated, truncated, info = env.step(action)

            env.render()

            thrust_cost = np.sum(np.abs(action))
            total_reward += float(reward)
            penalties += thrust_cost
            time_survived += 1

            if terminated:
                landed = True
                break

            if truncated:
                landed = False
                # break

            if debug:
                yield action, info

        x, y, vx, vy, angle, ang_vel, _, _ = observation

        values = np.array([
            # weight, value
            [1, total_reward],

            # Landing score
            [20, 1 / 1 + (x ** 2 + y ** 2)],
            [200, int(landed)],
            [10, 1 / (1 + (vx**2 + vy**2))],
            [-100, 1 / (1 + y)],

            [-1, penalties],
            [1, time_survived]
        ], dtype=np.float64)

        fitness = float(values[:, 0] @ values[:, 1])

        self.set_fitness(fitness)

def main():
    lander = Lander()
    print(lander.nn)

    env = args.make_env()

    for action, _ in lander.run(env=env, debug=True):
        print(f'action={np.array2string(action, precision=2)}', end='\r')
    else:
        print()

    env.close()

if __name__ == '__main__':
    main()
