import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
#RENDER_MODE = None #seleccione esta opção para não visualizar o ambiente (testes mais rápidos)
EPISODES = 1000

env = gym.make("LunarLander-v3", render_mode =RENDER_MODE,
    continuous=True, gravity=GRAVITY,
    enable_wind=ENABLE_WIND, wind_power=WIND_POWER,
    turbulence_power=TURBULENCE_POWER)


def check_successful_landing(observation):
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

def simulate(steps=1000,seed=None, policy = None):
    observ, _ = env.reset(seed=seed)
    for step in range(steps):
        action = policy(observ)

        observ, _, term, trunc, _ = env.step(action)

        if term or trunc:
            break

    success = check_successful_landing(observ)
    return step, success



#Perceptions
##TODO: Defina as suas perceções aqui

#Actions
##TODO: Defina as suas ações aqui
def get_actions(observation):
    #0,5 < mot_p < 1
    #-0,5 < mot_s dir (msd -> roda pr esq ) < -1 and 0,5 < mot_s esq (mse -> roda pr dir) < 1 
    #obs0 = x, obs1 = y, obs2 = vx, obs3 = vy, obs4 = theta, obs5 = vel_ang, obs6 = contact_left, obs7 = contact_right
    x = observation[0]
    y = observation[1]
    vx = observation[2]
    vy = observation[3]
    theta = observation[4]
    vel_ang = observation[5]

    action = [0,0] 

    if abs(x) > 0.2: 
        if x < -0.2:
            if vx > 0.2:
                pass
            if np.deg2rad(-5) > theta > np.deg2rad(-15): 
                action += np.array([0.6, 0.0])
            elif theta < np.deg2rad(-15):
                action += np.array([0.6, -0.6])
            elif theta > np.deg2rad(-5):
                action += np.array([0.6, 0.6])
        elif x > 0.2:
            if abs(vx) < -0.2:
                pass
            if np.deg2rad(5) <theta < np.deg2rad(15): 
                action += np.array([0.6, 0.0])
            elif theta > np.deg2rad(15):
                action += np.array([0.6, 0.6])
            elif theta < np.deg2rad(5):
                action += np.array([0.6, -0.6])
   
    else: 
        if abs(theta) > np.deg2rad(5):
            if theta < 0:
                action += np.array([0.0, -0.6])
            elif theta > 0:
                action += np.array([0.0, 0.6])
            else:
                pass 
        if vy < -0.4:
            action += np.array([0.6, 0.0])

    if vy > 0: 
        action += np.array([-1.0, 0.0])



    return action


def reactive_agent(observation):
    ##TODO: Implemente aqui o seu agente reativo
    ##Substitua a linha abaixo pela sua implementação
    action = env.action_space.sample()
    return action


def keyboard_agent(observation):
    action = [0,0]
    keys = pygame.key.get_pressed()

    print('observação:',observation)

    if keys[pygame.K_UP]:
        action =+ np.array([1,0])
    if keys[pygame.K_LEFT]:
        action =+ np.array( [0,-1])
    if keys[pygame.K_RIGHT]:
        action =+ np.array([0,1])

    return action


success = 0.0
steps = 0.0
for i in range(EPISODES):
    st, su = simulate(steps=1000000, policy=reactive_agent)
    if su:
        steps += st
    success += su

    if su>0:
        print('Média de passos das aterragens bem sucedidas:', steps/(su*(i+1))*100)
    print('Taxa de sucesso:', success/(i+1)*100)
