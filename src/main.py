from multiprocessing import Process, Queue
from typing import Tuple
import argparse
import copy
import gymnasium as gym
import numpy as np
import os
import random

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

env = gym.make(
    "LunarLander-v3",
    render_mode=args.render,
    continuous=True,
    gravity=args.gravity,
    enable_wind=args.wind,
    wind_power=args.wind_power,
    turbulence_power=args.turbulence_power,
)

NUM_PROCESSES = os.cpu_count()
evaluationQueue = Queue()
evaluatedQueue = Queue()

class NeuralNetwork:
    shape: tuple[int, int, int]
    GENOTYPE_SIZE: int

    SHAPE_TYPE = tuple[int, int, int]

    def __init__(self, n_inputs, n_outputs):
        self.shape = (n_inputs, 12, n_outputs)
        self.GENOTYPE_SIZE = NeuralNetwork.get_genotype_size(self.shape)

    @staticmethod
    def get_genotype_size(shape: SHAPE_TYPE) -> int:
        genotype_size = 0
        for i in range(1, len(shape)):
            genotype_size += shape[i - 1] * shape[i]

        return genotype_size

    def feed_forward(self, input: np.ndarray, ind: np.ndarray) -> np.ndarray:
        x = input.copy()
        for i in range(1, len(self.shape)):
            y = np.zeros(self.shape[i])
            for j in range(self.shape[i]):
                for k in range(len(x)):
                    y[j] += x[k] * ind[k + j * len(x)]
            # x = self.activation_function(y)
            x = np.tanh(y)
        return x

    def activation_function(self, number: float) -> float:
        return np.tanh(number);

    def simulate(
        self,
        *,
        steps: int = 50,
        render_mode = args.render,
        env = None,
        seed = None,
    ):
        # Simulates an episode of Lunar Lander, evaluating an individual
        env_was_none = env is None
        if env is None:
            env = gym.make(
                "LunarLander-v3",
                render_mode=render_mode,
                continuous=True,
                gravity=args.gravity,
                enable_wind=args.wind,
                wind_power=args.wind,
                turbulence_power=args.turbulence_power,
            )

        observation, _ = env.reset(seed=seed)

        prev_observation = observation
        for _ in range(steps):
            prev_observation = observation
            # Chooses an action based on the individual's genotype
            # HACK: genotype missing
            action = self.feed_forward(observation, np.random.rand(self.GENOTYPE_SIZE))  # Using random for demonstration
            observation, reward, terminated, truncated, info = env.step(action)
            # total_reward += reward

            if terminated == True or truncated == True:
                break

        if env_was_none:
            env.close()

        return self.objective_function(prev_observation)

    def objective_function(self, observation: np.ndarray):
        x = observation[0]
        y = observation[1]

        return (-abs(x) - abs(y), check_successful_landing(observation))

class Agent:
    nn: NeuralNetwork
    genotype: np.ndarray
    fitness: None | int

    def __init__(self):
        self.nn = NeuralNetwork(8, 2)  # Assuming input size 8 and output size 2 for lunar lander
        self.genotype = np.random.rand(self.nn.GENOTYPE_SIZE)
        self.fitness = None

class Population:
    def __init__(
        self,
        *,
        size: int = 50,
        generations: int = args.generations,
        crossover_probability: float = args.crossover_probability,
        mutation_probability: float = args.mutation_probability,
        elite_size: int = 1,
        nn_shape: Tuple[int, int, int] = (8, 12, 2),
    ):
        self.population_size = size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.nn_shape = nn_shape
        self.genotype_size = NeuralNetwork.get_genotype_size(nn_shape)
        self.population = []
        self.elite_size = elite_size

    def generate_initial_population(self):
        self.population = [Agent() for _ in range(self.population_size)]

    @staticmethod
    def evaluate_population(population):
        for agent in population:
            evaluationQueue.put(agent)
        new_pop = []
        for _ in population:
            ind = evaluatedQueue.get()
            new_pop.append(ind)
        return new_pop

    def parent_selection(self):
        return copy.deepcopy(random.choice(self.population))

    @staticmethod
    def crossover(p1: Agent, p2: Agent) -> Agent:
        # Perform crossover between p1 and p2
        child = Agent()
        crossover_point = random.randint(1, p1.nn.GENOTYPE_SIZE - 1)
        child.genotype[:crossover_point] = p1.genotype[:crossover_point]
        child.genotype[crossover_point:] = p2.genotype[crossover_point:]
        return child

    def mutation(self, agent: Agent) -> Agent:
        for i in range(agent.nn.GENOTYPE_SIZE):
            if random.random() < self.mutation_probability:
                agent.genotype[i] += np.random.normal(0, 0.1)  # Mutate slightly
        return agent

    def survival_selection(self, offspring: list[Agent]):
        self.population.sort(key=lambda agent: agent.fitness, reverse=True)
        new_population = self.population[:self.elite_size] + offspring
        new_population.sort(key=lambda agent: agent.fitness, reverse=True)
        self.population = new_population[:self.population_size]

    def evolution(self):
        # Create evaluation processes
        evaluation_processes = []
        assert NUM_PROCESSES is not None
        for _ in range(NUM_PROCESSES):
            p = Process(target=evaluate_worker, args=(evaluationQueue, evaluatedQueue))
            evaluation_processes.append(p)
            p.start()

        # Create initial population
        self.generate_initial_population()
        self.population = self.evaluate_population(self.population)

        # Iterate over generations
        for gen in range(self.generations):
            offspring = []
            while len(offspring) < self.population_size:
                if random.random() < self.crossover_probability:
                    p1 = self.parent_selection()
                    p2 = self.parent_selection()
                    child = self.crossover(p1, p2)
                else:
                    child = self.parent_selection()

                child = self.mutation(child)
                offspring.append(child)

            # Evaluate offspring
            offspring = self.evaluate_population(offspring)

            # Apply survival selection
            self.survival_selection(offspring)

            # Print best fitness of the current generation
            best_fitness = self.population[0].fitness
            print(f'Best of generation {gen}: {best_fitness}')

        # Stop evaluation processes
        assert NUM_PROCESSES is not None
        for _ in range(NUM_PROCESSES):
            evaluationQueue.put(None)
        for p in evaluation_processes:
            p.join()

def evaluate_worker(evaluationQueue, evaluatedQueue):
    env = gym.make(
        "LunarLander-v3",
        render_mode=args.render,
        continuous=True,
        gravity=args.gravity,
        enable_wind=args.wind,
        wind_power=args.wind_power,
        turbulence_power=args.turbulence_power
    )
    while True:
        ind = evaluationQueue.get()
        if ind is None:
            break

        ind.fitness = ind.nn.simulate(env=env)  # Assuming the neural network simulates itself
        evaluatedQueue.put(ind)

def check_successful_landing(observation: np.ndarray) -> bool:
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1
    on_landing_pad = abs(x) <= 0.2
    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)

    return legs_touching and on_landing_pad and stable_velocity and stable_orientation

def show_best_agent(population: Population):
    best_agent = population.population[0]
    env = gym.make(
        "LunarLander-v3",
        render_mode='human',
        gravity=args.gravity,
        enable_wind=args.wind,
        wind_power=args.wind_power,
        turbulence_power=args.turbulence_power
    )
    observation, _ = env.reset()

    while True:
        action = best_agent.nn.feed_forward(observation, best_agent.genotype)
        action = action.astype(np.float64).clip(-1., 1.)
        print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()  # Render the environment to the screen

        if terminated or truncated:
            break

    env.close()

def main():
    population = Population(nn_shape=(8, 12, 2))
    if args.evolve:
        population.evolution()
        show_best_agent(population)
    else:
        # Load bests from file (not yet implemented)
        pass

if __name__ == '__main__':
    main()
