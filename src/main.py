import random
import copy
import numpy as np
import gymnasium as gym 
import os
import argparse
import matplotlib.pyplot as plot
from multiprocessing import Process, Queue

# CONFIG
ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
EPISODES = 1000
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

POPULATION_SIZE = 50
NUMBER_OF_GENERATIONS = 10
PROB_CROSSOVER = 0.7

  
PROB_MUTATION = 1.0/GENOTYPE_SIZE
STD_DEV = 0.1


ELITE_SIZE = 1

TOURNAMENT_SIZE = 3

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
    # generation: 100
    # base: 1000
    default=100,
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
    default=True,
    help="Enable the wind."
)
parser.add_argument(
    '--generations',
    type=int,
    # base: 30
    default=20,
    help="The number of generations"
)
parser.add_argument(
    '--population',
    type=int,
    default=50,
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
    default=30,
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
    velocity_penalty = abs(vx) + abs(vy)  # Penalize high velocities
    angle_penalty = abs(theta)  # Penalize being tilted
    angular_vel_penalty = abs(omega)  # Penalize spinning
    
    # Landing bonuses
    leg_contact_bonus = 0
    if contact_left or contact_right:
        leg_contact_bonus = 0.5  # Partial bonus for one leg
    if contact_left and contact_right:
        leg_contact_bonus = 1.0  # Full bonus for both legs
        
    success = check_successful_landing(observation)
    landing_bonus = 10.0 if success else 0.0
    
    # Combine components with weights
    fitness = (
        -position_penalty * 0.5 + 
        -velocity_penalty * 0.3 + 
        -angle_penalty * 0.1 + 
        -angular_vel_penalty * 0.1 + 
        leg_contact_bonus * 0.5 + 
        landing_bonus
    )
    
    return fitness, success

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
    for i in range(2, tournament_size):
        next = gladiators[i]
        if next['fitness'] > best['fitness']:
            best = next
    return best

def parent_selection(population, tournament_size=TOURNAMENT_SIZE, number_of_parents=2):
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
            saved = population.pop(population.index(selected[i]))
        population.append(saved)
        return selected

def crossover(p1, p2):
    #TODO crossover com base no fitness -> p MAIS fit -> >% de genes 
    # Perform single-point crossover
    offspring = {'genotype': [], 'fitness': None}
    crossover_point = random.randint(0, GENOTYPE_SIZE - 1)
    offspring['genotype'] = (
        p1['genotype'][:crossover_point] + 
        p2['genotype'][crossover_point:]
    )
    # add the fitness value based on the parents
    offspring['fitness'] = (p1['fitness'] * crossover_point/GENOTYPE_SIZE + p2['fitness'] * 1 - crossover_point/GENOTYPE_SIZE) / 2
    
    return offspring

def mutation(p):
    #TODO num_gens com, base no fitness-> atraves de uma função 
    #Mutate the individual p
    if np.random.rand() < PROB_MUTATION:
        if (abs(p['fitness']) > 0.25):
            # If the individual is not successful, apply a stronger mutation
            mutations_points = random.sample(range(GENOTYPE_SIZE), 2)
            mutation_values = np.random.normal(0, STD_DEV , size=2)
            for i, mutation_point in enumerate(mutations_points):
                p['genotype'][mutation_point] += mutation_values[i]
                # Ensure the genotype values are within the range [-1, 1]
                p['genotype'][mutation_point] = np.clip(p['genotype'][mutation_point], -1, 1)
        else:
            # If the individual is successful, apply a weaker mutation
            mutation_point = random.randint(0, GENOTYPE_SIZE - 1)
            mutation_value = np.random.normal(0, STD_DEV)
            p['genotype'][mutation_point] += mutation_value
            # Ensure the genotype values are within the range [-1, 1]
            p['genotype'][mutation_point] = np.clip(p['genotype'][mutation_point], -1, 1)

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
                p1, p2 = parent_selection(population, number_of_parents=2)
                ni = crossover(p1, p2)

            else:
                ni = parent_selection(population, number_of_parents=1)
                
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

if __name__ == '__main__':
    render_mode = RENDER_MODE
    log_num = args.log
    if log_num == '-1':
        evolve = 1
    else: 
        evolve = 0
        #render_mode = 'human'
        
    if evolve:
        seeds = [964, 952, 364, 913, 140, 726, 112, 631, 881, 844, 965, 672, 335, 611, 457, 591, 551, 538, 673, 437, 513, 893, 709, 489, 788, 709, 751, 467, 596, 976]
        for i in range(NUMBER_OF_FILES):    
            random.seed(seeds[i])
            bests = evolution(i)
            with open(f'log{i}.txt', 'w') as f:
                for b in bests:
                    f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')
        
        plot.plot(results)
        plot.show()
        print(results)

                
    else:
        #validate individual
        bests = load_bests('log'+ log_num +'.txt')
        b = bests[-1]
        SHAPE = b[1]
        ind = b[2]
            
        ind = {'genotype': ind, 'fitness': None}
            
        #ntests = 1000    
        ntests = 100

        fit, success = 0, 0
        for i in range(1,ntests+1):
            f, s = simulate(ind['genotype'], render_mode=render_mode, seed = None)
            fit += f
            success += s
            
        print(fit/ntests, success/ntests)
