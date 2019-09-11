from unityremote.core import RemoteEnv
import numpy as np


env = RemoteEnv()
env.open()

actions = [('fx', 1.0), ('fx', -1.0), ('fy', 1.0), ('fy', -1.0), ('left_turn', 1.0), ('left_turn', -1), ('right_turn', 1), ('right_turn', -1), ('up',1),
            ('down', 1), ('push', True), ('push', False), ('crouch', True), ('crouch', False),
                ('pickup', True), ('pickup', False)]

action_size = len(actions)
    
for i in range(100000):
    idx = np.random.choice(action_size)
    env.step(actions[idx][0], actions[idx][1])

env.close()