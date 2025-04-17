from args import RenderMode, args
from lander import Lander
from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np
import random

class Population:
    landers: list[Lander]
    epochs: int
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
        generate_random_seed: bool = False,
        epochs: int = args.epochs,
    ) -> None:
        self.landers = [Lander() for _ in range(size)]
        self.generation = 1
        self.elite_size = 5
        self.seed = 12345 if not generate_random_seed else None
        self.env = args.make_env()
        self.epochs = epochs
        self.crossover_probability = args.crossover_probability
        self.mutation_probability = args.mutation_probability

    def __str__(self) -> str:
        return f'''
Population:
  size:       {len(self.landers)}
  epochs:     {self.epochs}
  elite size: {self.elite_size}
  crossover:  {self.crossover_probability * 100}%
  mutation:   {self.mutation_probability * 100}%
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
            for _ in lander.run(env=self.env, seed=self.seed):
                pass

        self.generation += 1

    def evolve(
        self,
        generations: int | None = None,
        *,
        debug: bool = False,
    ):
        for generation in range(1, (generations or self.epochs) + 1):
            if debug:
                print(f'{generation=}')
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

    def reproduce(self):
        """Create a new generation of landers from the elite."""
        elite_landers = self.select_elite()
        new_landers = []

        while len(new_landers) < len(self.landers):
            # TODO: Pick random 2 elite landers
            parent1, parent2 = random.sample(elite_landers, 2)
            child = self.crossover(parent1, parent2)
            child.mutate(self.mutation_probability)
            new_landers.append(child)

        self.landers = new_landers


def main():
    with Population(args.population_size, epochs=args.epochs) as population:
        print(population)

        if args.evolve:
            fitness_avgs = []

            plt.ion()
            plt.show()
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Fitness Average vs Generation')

            for fitness_array in population.evolve(debug=True):
                fitness_avgs.append(fitness_array.mean())
                print(f'Fitness values: {fitness_array[:5]}')

                x = np.linspace(1, len(fitness_avgs), len(fitness_avgs), dtype=np.int32)

                plt.xticks(x)
                plt.plot(fitness_avgs)

                plt.draw()
                plt.pause(.1)

            plt.ioff()
            plt.show()


        best = population.select_elite()[0]
        best.serialize('best.elite', append=True)
        env = args.make_env(render_mode=RenderMode.RenderHuman)

        for _ in range(5): # Number of runs
            for action, _ in best.run(env=env, debug=True):
                print(f'action={np.array2string(action, precision=2)}', end='\r')
            else:
                print()

        env.close()



if __name__ == '__main__':
    main()
