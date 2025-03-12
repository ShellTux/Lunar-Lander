from typing import Callable
import argparse
import gymnasium as gym
import numpy as np
import pygame
import random

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
    default=30,
    help="The number of generations"
)
parser.add_argument(
    '--specimen',
    type=int,
    default=10,
    help="The number of specimen"
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
    stable = stable_velocity and stable_orientation

    if legs_touching and on_landing_pad and stable:
        print("✅ Aterragem bem sucedida!")
        return True

    print("⚠️ Aterragem falhada!")
    return False

def simulate(
    steps: int = 1000,
    *,
    seed: int | None = None,
    policy: Callable[[np.ndarray], np.ndarray],
    current_specimen: np.array
) -> tuple[int, bool]:
    observ, _ = env.reset(seed=seed)
    step = 0
    for step in range(steps):
        action = policy(observ, current_specimen)

        observ, _, term, trunc, _ = env.step(action)

        if term or trunc:
            break

    success = check_successful_landing(observ)
    return step, success

def get_actions(
        observation: np.ndarray,
        current_specimen: np.array
    ) -> np.ndarray:
    x, y, vx, vy, theta, vel_ang, _, _  = observation

    theta_deg = np.rad2deg(theta)

    action: tuple[float, float] = (0, 0)

    if                                                                                 0 < vy                          : action = (-1,   0)
    elif      x < -.2 and -current_specimen[1] > theta_deg > -current_specimen[0]                                      : action = (.6,   0)
    elif      x < -.2 and                        theta_deg < -current_specimen[0]                                      : action = (.6, -.6)
    elif      x < -.2 and -current_specimen[1] < theta_deg                                                             : action = (.6,  .6)
    elif .2 < x       and  current_specimen[1] < theta_deg <  current_specimen[0]                                      : action = (.6,   0)
    elif .2 < x       and  current_specimen[0] < theta_deg                                                             : action = (.6,  .6)
    elif .2 < x       and                        theta_deg <   current_specimen[1]                                     : action = (.6, -.6)
    elif abs(x) < .2  and                        theta_deg <  -current_specimen[2]  and    vy <= -.5                   : action = (.6, -.6)
    elif abs(x) < .2  and  current_specimen[2] < theta_deg                          and    vy <= -.5                   : action = (.6,  .6)
    elif abs(x) < .2  and                        theta_deg <  -current_specimen[2]                                     : action = ( 0, -.6)
    elif abs(x) < .2  and  current_specimen[2] < theta_deg                                                             : action = ( 0,  .6)
    elif abs(x) < .2                                                                and    vy <  -current_specimen[3]  : action = (.6,   0)

    action_np = np.array(action, dtype=np.float64)

    return action_np

def reactive_agent(
        observation: np.ndarray,
        current_specimen: np.array
    ) -> np.ndarray:
    # TODO: Implemente aqui o seu agente reativo. Substitua a linha abaixo pela
    # sua implementação
    action = get_actions(observation, current_specimen)
    return action


def keyboard_agent(observation: np.ndarray) -> np.ndarray:
    keys = pygame.key.get_pressed()

    print(f'observação: {observation}')

    action = np.array([0, 0])
    if keys[pygame.K_UP]:
        action = np.array([1, 0])
    if keys[pygame.K_LEFT]:
        action = np.array([0, -1])
    if keys[pygame.K_RIGHT]:
        action = np.array([0, 1])

    return action



def main():
    # Sets up variables
    current_generations = np.zeros((args.specimen, 4))
    generation_best = np.zeros((3))
    generation_best_atributes = np.zeros((3, 4))
    generation_best_ind = np.zeros((3), dtype = np.uint16)
    results = np.zeros((args.specimen))
    max_success_rate = 0
    max_success_atributes = np.zeros((4))
    # Creates random starting atributes
    for i in range(args.specimen):
        random_traits = [(random.random()*10 + 10), 
                         (random.random()*10 + 1), 
                         (random.random()*5 + 1), 
                         (random.random()*0.5 - 1)]
        current_generations[i,:] = np.array(random_traits)
    # Starts testing for each generation
    for gen in range(args.generations):
        for spec in range(args.specimen):
            results[spec] = run_case_test(current_generations[spec,:])
            print('### Gen:', gen, 'spec: ', spec, 'Taxa de sucesso:', results[spec], '###')
            # if one of the new resulsts is better than the lowest, add it to the best's cases
            if (generation_best.min() < results[spec]):
                replaced = generation_best.argmin()
                generation_best_ind[replaced] = spec
                generation_best_atributes[replaced, :] = current_generations[spec,:]
                generation_best[replaced] = results[spec]
        # if the best case is inferior to one of the new generations, said generation becomes the new best
        if max_success_rate < generation_best.max():
            new_best = generation_best.argmax()
            # new_best_ind = generation_best_ind[new_best]
            # print(new_best_ind)
            max_success_atributes = current_generations[generation_best_ind[new_best], :]
            max_success_rate = generation_best[new_best]
        # ignore if it is the final generation
        if gen+1 >= args.generations:
            # mutate from the top 3 best specimens
            for passed_gen in range(3):
                for created_gen in range(3):
                    current_generations[created_gen+passed_gen*3, :] = np.array(
                        [generation_best_atributes[passed_gen, 0]+(random.random()-0.5)*results[passed_gen], 
                         generation_best_atributes[passed_gen, 1]+(random.random()*0.5-0.25)*results[passed_gen], 
                         generation_best_atributes[passed_gen, 2]+(random.random()*0.3-0.15)*results[passed_gen], 
                         generation_best_atributes[passed_gen, 3]+(random.random()*0.05-0.025)*results[passed_gen]])
            # get the average of the best 3
            current_generations[9, :] = np.array([
                np.mean(generation_best_atributes[:, 0]),
                np.mean(generation_best_atributes[:, 1]),
                np.mean(generation_best_atributes[:, 2]),
                np.mean(generation_best_atributes[:, 3])])
    print("Best success rate: ", max_success_rate)
    print("Best atributes: ", max_success_atributes)


def run_case_test(current_specimen: np.array):
    success = 0.0
    steps = 0.0
    for i in range(args.episodes):
        st, su = simulate(steps=1000000, policy=reactive_agent, current_specimen=current_specimen)
        if su:
            steps += st
        success += su

        if su > 0:
            print('Média de passos das aterragens bem sucedidas:', steps/(su*(i+1))*100)
        print('Taxa de sucesso:', success/(i+1)*100)
    return success/(args.episodes)*100

if __name__ == '__main__':
    main()
