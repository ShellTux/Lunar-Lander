from typing import Callable
import argparse
import gymnasium as gym
import numpy as np
import pygame

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
    policy: Callable[[np.ndarray], np.ndarray]
) -> tuple[int, bool]:
    observ, _ = env.reset(seed=seed)
    step = 0
    for step in range(steps):
        action = policy(observ)

        observ, _, term, trunc, _ = env.step(action)

        if term or trunc:
            break

    success = check_successful_landing(observ)
    return step, success

def get_actions(observation: np.ndarray) -> np.ndarray:
    x, y, vx, vy, theta, vel_ang, _, _  = observation

    theta_deg = np.rad2deg(theta)

    action: tuple[float, float] = (0, 0)

    if                                             0 < vy       : action = (-1,   0)
    elif      x < -.2 and -5 > theta_deg > -15                  : action = (.6,   0)
    elif      x < -.2 and      theta_deg < -15                  : action = (.6, -.6)
    elif      x < -.2 and -5 < theta_deg                        : action = (.6,  .6)
    elif .2 < x       and  5 < theta_deg <  15                  : action = (.6,   0)
    elif .2 < x       and 15 < theta_deg                        : action = (.6,  .6)
    elif .2 < x       and      theta_deg <   5                  : action = (.6, -.6)
    elif abs(x) < .2  and      theta_deg <  -5 and     vy <= -.5: action = (.6, -.6)
    elif abs(x) < .2  and  5 < theta_deg       and     vy <= -.5: action = (.6,  .6)
    elif abs(x) < .2  and      theta_deg <  -5                  : action = ( 0, -.6)
    elif abs(x) < .2  and  5 < theta_deg                        : action = ( 0,  .6)
    elif abs(x) < .2                           and     vy <  -.4: action = (.6,   0)

    action_np = np.array(action, dtype=np.float64)

    return action_np

def reactive_agent(observation: np.ndarray) -> np.ndarray:
    # TODO: Implemente aqui o seu agente reativo. Substitua a linha abaixo pela
    # sua implementação
    action = get_actions(observation)
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
    success = 0.0
    steps = 0.0
    for i in range(args.episodes):
        st, su = simulate(steps=1000000, policy=reactive_agent)
        if su:
            steps += st
        success += su

        if su > 0:
            print('Média de passos das aterragens bem sucedidas:', steps/(su*(i+1))*100)
        print('Taxa de sucesso:', success/(i+1)*100)

if __name__ == '__main__':
    main()
