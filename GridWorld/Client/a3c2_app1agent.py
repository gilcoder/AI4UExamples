from unityremote.ml.a3c.train import run as run_train
from unityremote.ml.a3c.run_checkpoint import run as run_test
from unityremote.utils import environment_definitions
import UnityRemoteGym
import numpy as np
import argparse
from gym.core import Wrapper
from unityremote.utils import image_from_str
from collections import deque
from matplotlib import pyplot as plt


IMAGE_SHAPE = (10, 10, 1)
ARRAY_SIZE = 4
ACTION_SIZE = 5

def make_inference_network(obs_shape, n_actions, debug=False, extra_inputs_shape=None):
    import tensorflow as tf
    from unityremote.ml.a3c.multi_scope_train_op import make_train_op 
    from unityremote.ml.a3c.utils_tensorflow import make_grad_histograms, make_histograms, make_rmsprop_histograms, \
        logit_entropy, make_copy_ops

    observations = tf.placeholder(tf.float32, [None] + list(obs_shape))
    proprioceptions = tf.placeholder(tf.float32, (None, ARRAY_SIZE) )
    
    normalized_obs = tf.keras.layers.Lambda(lambda x : x/6.0)(observations)

    # Numerical arguments are filters, kernel_size, strides
    conv1 = tf.keras.layers.Conv2D(16, (1,1), (1,1), activation='relu', name='conv1')(normalized_obs)
    if debug:
        # Dump observations as fed into the network to stderr for viewing with show_observations.py.
        conv1 = tf.Print(conv1, [observations], message='\ndebug observations:',
                         summarize=2147483647)  # max no. of values to display; max int32
    
    conv2 = tf.keras.layers.Conv2D(16, (3,3), (1,1), activation='relu', name='conv2')(conv1)
    #conv3 = tf.layers.conv2d(conv2, 16, 3, 1, activation=tf.nn.relu, name='conv3')

    #phidden = tf.keras.layers.Dense(30, activation='relu', name='phidden')(proprioceptions[:, 0:ARRAY_SIZE])
    phidden = tf.keras.layers.Dense(30, activation='relu', name='phidden')(proprioceptions)
    phidden2 = tf.keras.layers.Dense(30, activation='relu', name='phidden2')(phidden)

    flattened = tf.keras.layers.Flatten()(conv2)

    expanded_features = tf.keras.layers.Concatenate()([flattened, phidden2])

    hidden = tf.keras.layers.Dense(512, activation='relu', name='hidden')(expanded_features)
    #hidden2 = tf.keras.layers.Lambda(lambda x: x * proprioceptions[:,9:10])(hidden)
    #action_logits = tf.keras.layers.Dense(n_actions, activation=None, name='action_logits')(hidden2)
    action_logits = tf.keras.layers.Dense(n_actions, activation=None, name='action_logits')(hidden)
    action_probs = tf.nn.softmax(action_logits)
    #values = tf.layers.Dense(1, activation=None, name='value')(hidden2)
    values = tf.layers.Dense(1, activation=None, name='value')(hidden)


    # Shape is currently (?, 1)
    # Convert to just (?)
    values = values[:, 0]

    layers = [conv1, conv2, phidden, phidden2, flattened, expanded_features, hidden]

    return (observations, proprioceptions), action_logits, action_probs, values, layers



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",
                        choices=['train', 'test'],
                        default='train')
    parser.add_argument('--path', default='.')
    parser.add_argument('--preprocessing', choices=['generic', 'user_defined'])
    return parser.parse_args()

def get_frame_from_fields(fields):
    imgdata = image_from_str(fields['state'], 10, 10)
    return imgdata


MAX_ENERGY = [20, 100, 1000, 300, 1000]
MIN_ENERGY = [10, 50, 10, 200, 100 ]
ENERGY_DECAY = 0.99
ENERGY_REST = 0.09

ENERGY_CAP = 500.0

class Agent:
    def __init__(self):
        self.goal_checker = goal_checker()
        self.energy = MIN_ENERGY[self.goal_checker.g] + (MAX_ENERGY[self.goal_checker.g]-MIN_ENERGY[self.goal_checker.g])/2.0
        #self.goal = np.random.choice([0, 1])
        self.goal = 1
        self.suc = 0
        self.total = 0
        self.ep = 0
        self.pltmin = []
        self.pltmax = []
        self.pltvalue = []
        self.pgoal = -1

    def __make_state__(self, frame, env_info):        
        frame = frame.reshape(1, 10, 10)
        frame = np.moveaxis(frame, 0, -1)
        
        done = env_info['done']
        reward = env_info['reward']
        
        reward = np.clip(reward, -1, +1)

        proprioceptions = np.zeros(ARRAY_SIZE)

        proprioceptions[0] = self.energy/ENERGY_CAP
        proprioceptions[1] = MIN_ENERGY[self.goal_checker.g]/ENERGY_CAP
        proprioceptions[2] = MAX_ENERGY[self.goal_checker.g]/ENERGY_CAP
        proprioceptions[3] = 0

        self.pltmin.append(proprioceptions[1])
        self.pltmax.append(proprioceptions[2])
        self.pltvalue.append(proprioceptions[0])
        prop = proprioceptions[:]
        return (frame, prop)


    def reset(self, env):
        self.goal = 1
        self.goal_checker.reset()
        #self.goal  = np.random.choice([0,1])
        self.goal = 1
        self.energy = MIN_ENERGY[self.goal_checker.g] + (MAX_ENERGY[self.goal_checker.g]-MIN_ENERGY[self.goal_checker.g])/2.0
        self.total = 0
        self.suc = 0
        self.ep += 1
        env_info = env.remoteenv.step("restart")
        return self.__make_state__(get_frame_from_fields(env_info), env_info)

    def act(self, env, action=None, info=None):
        
        for _ in range(8):
            env_info = env.one_step(action)

        frame = get_frame_from_fields(env_info)

        if action >= 0:
            self.energy = ENERGY_DECAY * self.energy;
        else:
            self.energy += ENERGY_REST * MAX_ENERGY[self.goal_checker.g];

        if self.energy > ENERGY_CAP:
            self.energy = ENERGY_CAP

        done = env_info['done']
        reward = env_info['reward'];
        
        reward = np.clip(reward, -1, +1)



        state = self.__make_state__(frame, env_info)

        if state[1][0] >= state[1][1] and state[1][0] <= state[1][2]:
            self.suc += 1

        self.total += 1

        reward, goal = self.goal_checker(self.goal, state[1][0], state[1][1], state[1][2], reward)
        
        state[1][3] = goal
        
        if self.pgoal >= 0 and self.pgoal != goal:
            done = True
        self.pgoal = goal

        if done:
            #print("Second goal success rate ", self.suc/self.total)
            if len(self.pltmax) > 0:
                fig = plt.figure()
                fig.suptitle('Agent Profile')
                plt.xlabel('Time Step')
                plt.ylabel('Energy Level')
                plt.plot(self.pltmax, label='max')
                plt.plot(self.pltmin, label='min')
                plt.plot(self.pltvalue, label='energy level')
                plt.legend(loc="upper left")
                fig.savefig('episode%d'%(self.ep))

            self.pltmax = []
            self.pltmin = []
            self.pltvalue = []

        return (state, reward, done, env_info)


class goal_checker:
    def __init__(self):
        self.g = np.random.choice(len(MAX_ENERGY))
        self.current_pos = self.g

    def reset(self):
        self.current_pos = 0 
        self.g = np.random.choice(len(MAX_ENERGY))
        self.current_pos =  MIN_ENERGY[self.g] + (MAX_ENERGY[self.g]-MIN_ENERGY[self.g])/2.0

    def __call__(self, goal, g, vmin, vmax, default_value=0):
        reward = default_value


        if (self.current_pos < vmin or self.current_pos > vmax) and (g >= vmin and g <= vmax):
            reward += 0.01
        elif (self.current_pos >= vmin and self.current_pos <= vmax) and (g < vmin or g > vmax):
            reward += -0.01
        else:
            ref = vmin + (vmax - vmin)/2.0

            d1 = abs(self.current_pos - ref)
            self.current_pos = g

            d2 = abs(g - ref)

            if d2 < d1:
                reward += 0.01
            elif d2 > d1:
                reward += -0.01

        if g >= vmin and g <= vmax:
            goal = 0
        elif g < vmin:
            goal = -1
        else:
            goal = 1

        return (reward, goal)

def make_env_def():
        environment_definitions['state_shape'] = IMAGE_SHAPE
        environment_definitions['action_shape'] = (ACTION_SIZE,)
        environment_definitions['actions'] = [('move',0), ('move', 1), ('move', 2), ('move', 3), ('NOOP', -1)]
        environment_definitions['action_meaning'] = ['forward', 'right', 'backward', 'left',  'NOOP']
        environment_definitions['agent'] = Agent
        environment_definitions['extra_inputs_shape'] = (ARRAY_SIZE,)
        environment_definitions['make_inference_network'] = make_inference_network

def train():
        args = ['--n_workers=8', '--steps_per_update=30', 'UnityRemote-v0']
        make_env_def()
        run_train(environment_definitions, args)

def test(path):
        args = ['UnityRemote-v0', path]
        make_env_def()
        run_test(environment_definitions, args)


if __name__ == '__main__':
   args = parse_args()
   if args.run == "train":
        train()
   elif args.run == "test":
        test(args.path)
