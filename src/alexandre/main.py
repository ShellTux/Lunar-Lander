from multiprocessing import Process, Queue
import argparse
import gymnasium as gym
import matplotlib.pyplot as plot
import numpy as np
import os
import random

# CONFIG
ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0

STEPS = 500

NUM_PROCESSES = os.cpu_count()
evaluationQueue = Queue()
evaluatedQueue = Queue()


nInputs = 8
nOutputs = 2
SHAPE = (nInputs,12,nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]

# PROB_CROSSOVER = 0.7
PROB_CROSSOVER = 0.5


# PROB_MUTATION = 1.0/GENOTYPE_SIZE   #probability of mutation for each gene
PROB_MUTATION = 0.008
STD_DEV = 0.1


ELITE_SIZE = 0

TOURNAMENT_SIZE = 5

parser = argparse.ArgumentParser(description='Lunar Lander')
parser.add_argument(
    '--render',
    choices=['none', 'human'],
    default='none',
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
    '--generations',
    type=int,
    # base: 30
    default=100,
    help="The number of generations"
)
parser.add_argument(
    '--population',
    type=int,
    default=100,
    help="The population size"
)
parser.add_argument(
    '--log',

    type=str,
    default='-1',
    help="The log file number, change the mod to load the bests"
)
parser.add_argument(
    '--nb_files',
    type=int,
    default=5,
    help="The number of files to generate"
)
args = parser.parse_args()
if args.render == 'human':
    RENDER_MODE = 'human'
elif args.render == 'none':
    RENDER_MODE = None
if args.episodes:
    EPISODES = args.episodes
if args.population:
    POPULATION_SIZE = args.population
if args.gravity:
    GRAVITY = args.gravity
if args.turbulence_power:
    TURBULENCE_POWER = args.turbulence_power
if args.wind_power:
    WIND_POWER = args.wind_power
if args.wind:
    ENABLE_WIND = True
if args.generations:
    NUMBER_OF_GENERATIONS = args.generations
if args.nb_files:
    NUMBER_OF_FILES = args.nb_files
results = np.zeros(NUMBER_OF_FILES)
# END OF CONFIG

def network(shape, observation,ind):
    #Computes the output of the neural network given the observation and the genotype
    x = observation[:]
    for i in range(1,len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k]*ind[k+j*len(x)]
        x = np.tanh(y)
    return x


def check_successful_landing(observation):
    #Checks the success of the landing based on the observation
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1

    on_landing_pad = abs(x) <= 0.2

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation

    if legs_touching and on_landing_pad and stable:
        return True
    return False

def objective_function(observation):
    # Computes the quality of the individual based on various factors
    x = observation[0]  # Horizontal distance to the landing pad
    y = observation[1]  # Vertical distance to the ground
    vx = observation[2]  # Horizontal velocity
    vy = observation[3]  # Vertical velocity
    theta = observation[4]  # Angle of the lander
    omega = observation[5]  # Angular velocity
    contact_left = observation[6]  # Left leg contact
    contact_right = observation[7]  # Right leg contact

    # Calculate individual components
    position_penalty = abs(x)  # Penalize being far from center
    velocity_penalty_Y =  abs(vy)  # Penalize high velocities
    velocity_penalty_X = abs(vx)  # Penalize high horizontal velocities
    angle_penalty = abs(theta)  # Penalize being tilted
    angular_vel_penalty = abs(omega)  # Penalize spinning

    success = check_successful_landing(observation)
    landing_bonus = 100.0 if success else 0.0
    # fitness = 100
    # Combine components with weights
    fitness = 100 - position_penalty*100
    if (position_penalty < 0.2):
        fitness += 100 - min(velocity_penalty_Y*20, 100)
        if (velocity_penalty_Y < 0.2):
            fitness += 100 - min(angle_penalty*20, 100)
            if (angle_penalty < 10):
                if (contact_left):
                    fitness += 50
                if (contact_right):
                    fitness += 50

    return fitness , success


def simulate(genotype, render_mode = None, seed=None, env = None):
    #Simulates an episode of Lunar Lander, evaluating an individual
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode =render_mode,
        continuous=True, gravity=GRAVITY,
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER,
        turbulence_power=TURBULENCE_POWER)

    observation, info = env.reset(seed=seed)

    for _ in range(STEPS):
        prev_observation = observation
        #Chooses an action based on the individual's genotype
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated == True or truncated == True:
            break

    if env_was_none:
        env.close()

    return objective_function(prev_observation)

def evaluate(evaluationQueue, evaluatedQueue):
    #Evaluates individuals until it receives None
    #This function runs on multiple processes

    env = gym.make("LunarLander-v3", render_mode =None,
        continuous=True, gravity=GRAVITY,
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER,
        turbulence_power=TURBULENCE_POWER)
    while True:
        ind = evaluationQueue.get()

        if ind is None:
            break

        ind['fitness'] = simulate(ind['genotype'], seed = None, env = env)[0]

        evaluatedQueue.put(ind)
    env.close()

def evaluate_population(population):
    #Evaluates a list of individuals using multiple processes
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop

def generate_initial_population():
    #Generates the initial population
    population = []
    for i in range(POPULATION_SIZE):
        #Each individual is a dictionary with a genotype and a fitness value
        #At this time, the fitness value is None
        #The genotype is a list of floats sampled from a uniform distribution between -1 and 1

        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population

def tournament(gladiators , tournament_size=TOURNAMENT_SIZE):
    best = gladiators[0]
    for i in range(1, tournament_size):
        next = gladiators[i]
        if next['fitness'] > best['fitness']:
            best = next
    return best

def parent_selection_tournament(population, tournament_size=TOURNAMENT_SIZE, number_of_parents=2):
    if len(population) < number_of_parents:
        raise ValueError("Not enough individuals in the population to select parents.")
    elif number_of_parents == 1:
        return random.sample(population, 1)[0]
    else:
        #TODO   tournment selection
        selected = []
        saved = []
        for i in range(number_of_parents):
            gladiators = random.sample(population, tournament_size)
            selected.append(tournament(gladiators))
            #saved = population.pop(population.index(selected[i]))
        #population.append(saved)
        return selected



def crossover(p1, p2):
    # Implement two-point crossover
    offspring = {'genotype': [], 'fitness': None}
    point1 = random.randint(0, GENOTYPE_SIZE - 2)
    point2 = random.randint(point1 + 1, GENOTYPE_SIZE - 1)

    # Create offspring genotype by combining segments from parents
    offspring['genotype'] = (
        p1['genotype'][:point1] +
        p2['genotype'][point1:point2] +
        p1['genotype'][point2:]
    )

    # Calculate fitness as weighted average based on gene segments inherited
    len_p1 = point1 + (GENOTYPE_SIZE - point2)
    len_p2 = point2 - point1
    offspring['fitness'] = (
        (p1['fitness'] * len_p1 + p2['fitness'] * len_p2) / GENOTYPE_SIZE
        if p1['fitness'] is not None and p2['fitness'] is not None else None
    )

    return offspring


def mutation(p):
    #TODO num_gens com, base no fitness-> atraves de uma função
    #Mutate the individual

    for i in range(GENOTYPE_SIZE):
        if np.random.rand() < PROB_MUTATION:
            mutation_value = np.random.normal(0, STD_DEV)
            p['genotype'][i] += mutation_value
            # Ensure the genotype values are within the range [-1, 1]
            p['genotype'][i] = np.clip(p['genotype'][i], -1, 1)


    return p

def survival_selection(population, offspring):
    #reevaluation of the elite
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population

def evolution(index):
    best_generational_results = np.zeros(NUMBER_OF_GENERATIONS)
    #Create evaluation processes
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluationQueue, evaluatedQueue)))
        evaluation_processes[-1].start()

    #Create initial population
    bests = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key = lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)

    #Iterate over generations
    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []

        #create offspring
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                p1, p2 = parent_selection_tournament(population, number_of_parents=2)
                ni = crossover(p1, p2)

            else:
                ni = parent_selection_tournament(population, number_of_parents=1)
            ni = mutation(ni)
            offspring.append(ni)

        #Evaluate offspring
        offspring = evaluate_population(offspring)

        #Apply survival selection
        population = survival_selection(population, offspring)

        #Print and save the best of the current generation
        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)
        best_generational_results[gen] = population[0]['fitness']
        print(f'Best of generation {gen}: {best[1]}')
    plot.title(f'Creation {index}')
    plot.plot(best_generational_results)
    plot.show()

    #Stop evaluation processes
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()

    #Return the list of bests
    results[index] = best[1]
    return bests

def load_bests(fname):
    #Load bests from file
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append(( eval(fitness),eval(shape), eval(genotype)))
    return bests

def plot_evolution(all_fits, all_sucs):
    # Convert boolean success values to integers for better visualization
    all_sucs_int = [100 if s else 0 for s in all_sucs]

    # Plot both fitness and success values on the same plot
    generations = list(range(len(all_fits)))
    plot.figure(figsize=(10, 6))
    plot.plot(generations, all_fits, marker='o', linestyle='-', label='Fitness')
    plot.plot(generations, all_sucs_int, marker='x', linestyle='--', label='Success')
    plot.title('Evolution of Fitness and Success Over Tests')
    plot.xlabel('Test Number')
    plot.ylabel('Value')
    plot.legend()
    plot.grid(True)
    plot.savefig('evolution_plot_combined.png')
    plot.show()

if __name__ == '__main__':
    # render_mode = RENDER_MODE
    render_mode = 'human'
    file_results = np.zeros(30)
    # log_num = args.log
    log_num = '3'
    # if log_num == '-1':
    #     evolve = 1
    # else:
    #     evolve = 0
    #     #render_mode = 'human'

    # # if evolve:
    # seeds = [964, 952, 364, 913, 140, 726, 112, 631, 881, 844, 965, 672, 335, 611, 457, 591, 551, 538, 673, 437, 513, 893, 709, 489, 788, 709, 751, 467, 596, 976]
    # for i in range(NUMBER_OF_FILES):
    #     random.seed(seeds[i])
    #     bests = evolution(i)
    #     with open(f'log{i}.txt', 'w') as f:
    #         for b in bests:
    #             f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')
    # plot.plot(results)
    # plot.show()
    # print(results)


    # else:
    # for log in range(NUMBER_OF_FILES):
    #     #validate individual
    #     log_str = str(log)
    #     bests = load_bests('log'+ log_str +'.txt')
    #     b = bests[-1]
    #     SHAPE = b[1]
    #     ind = b[2]

    #     ind = {'genotype': ind, 'fitness': None}

    #     ntests = 1000
    #     # ntests = 100

    #     fit, success = 0, 0
    #     all_fits, all_sucs = [], []
    #     for i in range(1,ntests+1):
    #         f, s = simulate(ind['genotype'], render_mode=render_mode, seed = None)
    #         fit += f
    #         all_fits.append(f)
    #         success += s
    #         all_sucs.append(s)

    #     print(all_sucs.count(True))
    #     plot_evolution(all_fits, all_sucs)
    #     print(fit/ntests, success/ntests)
    #     file_results[log] = (success/ntests)
    # print(file_results)

    # log_str = str(log)
    bests = load_bests('log'+ log_num +'.txt')
    b = bests[-1]
    SHAPE = b[1]
    ind = b[2]

    ind = {'genotype': ind, 'fitness': None}

    ntests = 1000
    # ntests = 100

    fit, success = 0, 0
    all_fits, all_sucs = [], []
    for i in range(1,ntests+1):
        f, s = simulate(ind['genotype'], render_mode=render_mode, seed = None)
        fit += f
        all_fits.append(f)
        success += s
        all_sucs.append(s)

    print(all_sucs.count(True))
    plot_evolution(all_fits, all_sucs)
    print(fit/ntests, success/ntests)
