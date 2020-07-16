from ai4u.core import RemoteEnv
import numpy as np
from ai4u.utils import image_decode

def agent():
    env = RemoteEnv(IN_PORT=8085, OUT_PORT=7075)
    env.open(0)
    for i in range(10000000):
        state = env.step("restart")
        prev_energy = state['energy']
        done = state['done']
        while not done:
            action = int(input('action'))
            #action = np.random.choice([0, 1])
            reward_sum = 0
            touched = np.zeros(8)
            for i in range(8):
                state = env.step("act", action)
                state = env.step('get_status')
                energy = state['energy']
                touched[i] = state['touched']
                if state['done']:
                    break
            done = state['done']
            reward_sum += (energy - prev_energy)
            prev_energy = energy
            frame = state['frame']
            ID = state['id']
            frame = image_decode(state['frame'], 20, 20)
            print(frame)
            print(reward_sum)
            print(touched)
            print(state['signal'])
            print(state['done'])
            prev_energy = energy
        if i >= 2:
            break
    env.close()

agent()