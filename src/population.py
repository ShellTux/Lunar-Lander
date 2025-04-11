import random
from nn import NeuralNetwork
from args import args
import gymnasium as gym
import numpy as np
import time

class Lander:
    nn: NeuralNetwork
    fitness: float

    def __init__(self) -> None:
        # Initialize the neural network for the lander agent
        self.nn = NeuralNetwork(8, 6, 6, 2)
        self.fitness = 0  # Track the fitness score of the lander

    def set_fitness(self, fitness: float) -> None:
        """Set the fitness score of the lander."""
        self.fitness = fitness

    def run(
        self,
        *,
        env: gym.Env,
        debug: bool = False,
        seed: int | None = None,
    ) -> None:
        observation, _ = env.reset(seed=seed)

        total_reward: float = 0
        time_survived: int = 0
        penalties: float = 0

        while True:
            action = self.nn.forward(observation)
            # action = np.random.uniform(-1, 1, size=(2,)).astype(np.float32).clip(-1, 1).round(decimals=2)
            if debug:
                print(f'action={np.array2string(action, precision=2)}', end='\r')

            observation, reward, terminated, truncated, info = env.step(action)
            if debug and len(info) > 0:
                print(info)

            env.render()

            thrust_cost = np.sum(np.abs(action))
            total_reward += reward
            penalties += thrust_cost
            time_survived += 1

            if terminated or truncated:
                break

        if debug:
            print()

        distance_to_landing_pad = np.linalg.norm(observation[:2])
        landing_score = 1 / (1 + distance_to_landing_pad)

        fitness = total_reward + landing_score - penalties + time_survived

        self.set_fitness(fitness)

class Population:
    landers: list[Lander]
    generation: int
    elite_size: int
    seed: int | None
    env: gym.Env
    crossover_probability: float
    mutation_probability: float

    def __init__(
        self,
        size: int,
        *,
        generate_random_seed: bool = False
    ) -> None:
        self.landers = [Lander() for _ in range(size)]
        self.generation = 1
        self.elite_size = 5
        self.seed = 12345 if not generate_random_seed else None
        self.env = gym.make(
            "LunarLander-v3",
            render_mode=args.render,
            continuous=True,
            gravity=args.gravity,
            enable_wind=args.wind,
            wind_power=args.wind_power,
            turbulence_power=args.turbulence_power
        )
        self.crossover_probability = args.crossover_probability
        self.mutation_probability = args.mutation_probability

    def __str__(self) -> str:
        return f'''
Population:
  size: {len(self.landers)}
        '''.strip()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.env.close()

        if exc_type is not None:
            print(f"An exception occurred: {exc_value}")
        return False

    def run(self) -> None:
        for lander in self.landers:
            lander.run(env=self.env, seed=self.seed)

        self.generation += 1

    def evolve(
        self,
        generations: int,
        *,
        debug: bool = False,
    ):
        for _ in range(generations):
            if debug:
                print(f'{self.generation=}')
            self.run()

            self.landers.sort(key=lambda lander: lander.fitness, reverse=True)

            fitness_values = np.array([lander.fitness for lander in self.landers])

            yield fitness_values

            self.reproduce()

    def select_elite(self):
        """Select the elite landers based on fitness."""
        return sorted(self.landers, key=lambda lander: lander.fitness, reverse=True)[:self.elite_size]

    def crossover(self, parent1: Lander, parent2: Lander):
        """Create a new Lander by crossing over the weights of two parents."""
        child = Lander()

        # Assuming weights are stored in 'nn.weights'
        # TODO: Add more diversity by picking weights
        for i in range(len(child.nn.weights)):
            if np.random.rand() < self.crossover_probability:
                child.nn.weights[i] = (parent1.nn.weights[i] + parent2.nn.weights[i]) / 2
            else:
                child.nn.weights[i] = parent1.nn.weights[i].copy()

        return child

    def mutate(self, lander: Lander):
        """Mutate the weights of the Lander with a certain mutation rate."""
        for i in range(len(lander.nn.weights)):
            if np.random.rand() > self.mutation_probability:
                continue

            mutation_amount: np.ndarray = np.random.normal(0, .1, size=lander.nn.weights[i].shape)
            lander.nn.weights[i] = (lander.nn.weights[i] + mutation_amount).clip(-1, 1)

    def reproduce(self):
        """Create a new generation of landers from the elite."""
        elite_landers = self.select_elite()
        new_landers = []

        while len(new_landers) < len(self.landers):
            # TODO: Pick random 2 elite landers
            parent1, parent2 = random.sample(elite_landers, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_landers.append(child)

        self.landers = new_landers


def main():
    with Population(50) as population:
        print(population)

        for fitness_array in population.evolve(100, debug=True):
            print(f'Fitness values: {fitness_array[:5]}')

        best = population.select_elite()[0]
        env = gym.make(
            "LunarLander-v3",
            render_mode='human',
            continuous=True,
            gravity=args.gravity,
            enable_wind=args.wind,
            wind_power=args.wind_power,
            turbulence_power=args.turbulence_power
        )

        observation, _ = env.reset()

        while True:
            action = best.nn.forward(observation)
            print(f'action={np.array2string(action, precision=2)}', end='\r')

            observation, _, terminated, truncated, _ = env.step(action)
            env.render()

            if terminated or truncated:
                break

        print()
        env.close()


if __name__ == '__main__':
    main()
