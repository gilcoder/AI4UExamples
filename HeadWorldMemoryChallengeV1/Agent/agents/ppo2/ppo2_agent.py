import gym

from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from ai4u.utils import environment_definitions
import AI4UGym
from AI4UGym import BasicAgent
import numpy as np
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.deepq.policies import FeedForwardPolicy
from ai4u.utils import image_decode
from math import log, e
from collections import deque

def modified_cnn(unscaled_images, **kwargs):
	import tensorflow as tf
	scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
	activ = tf.nn.relu
	layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
	layer_2 = activ(conv(layer_1, 'c2', n_filters=32, filter_size=2, stride=2, init_scale=np.sqrt(2), **kwargs))
	layer_2 = conv_to_fc(layer_2)
	return activ(linear(layer_2, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

class CustomPolicy(CnnLstmPolicy):
	def __init__(self, *args, **kwargs):
		super(CustomPolicy, self).__init__(*args, **kwargs, n_lstm=64, cnn_extractor=modified_cnn)

TOUCH_SIZE = 4
IMAGE_SHAPE = (21, 20, 1)
ARRAY_SIZE = 20
ACTION_SIZE = 5

model = None

def make_env_def():
	environment_definitions['state_shape'] = IMAGE_SHAPE
	environment_definitions['action_shape'] = (ACTION_SIZE,)
	environment_definitions['actions'] = [('act', 0), ('act', 1), ('act', 3), ('act', 4), ('act', -1)]
	environment_definitions['input_port'] = 8080
	environment_definitions['output_port'] = 7070
	environment_definitions['agent'] = Agent
	environment_definitions['host'] = '127.0.0.1'
	BasicAgent.environment_definitions = environment_definitions


def get_frame_from_fields(frame, touchs):
	imgdata = image_decode(frame, 20, 20, np.uint8)
	proprioceptions = np.zeros(ARRAY_SIZE)
	for i in range(TOUCH_SIZE):
		proprioceptions[i] = touchs[i]
	proprioception = np.array(proprioceptions, dtype=np.uint8)
	imgdata = np.vstack([imgdata, proprioception])
	return imgdata

class Agent(BasicAgent):
	def __init__(self):
		BasicAgent.__init__(self)
		self.hist = deque(maxlen=30)

	def __make_state__(imageseq):
		frameseq = np.array([imageseq], dtype=np.uint8)
		frameseq = np.moveaxis(frameseq, 0, -1)
		return frameseq

	def reset(self, env):
		env_info = env.remoteenv.step("restart")
		for i in range(np.random.choice(30)):
			env.one_step(np.random.choice([0, 1, 3, 4]))
		return Agent.__make_state__( get_frame_from_fields(env_info['frame'], np.zeros(8)) )

	def act(self, env, action, info=None):
		touch_sensor = np.zeros(TOUCH_SIZE)
		sum_reward = 0
		for f in range(TOUCH_SIZE):
			envinfo = env.one_step(action)
			touch_sensor[f] = envinfo['touched']
			sum_reward += envinfo['reward']
			if envinfo['done']:
				break
		frame = get_frame_from_fields(envinfo['frame'], touch_sensor)
		state = Agent.__make_state__(frame)
		return state, sum_reward, envinfo['done'], envinfo


def train():
	make_env_def()
	# multiprocess environment
	env = make_vec_env('AI4U-v0', n_envs=8)
	model = PPO2(CustomPolicy, env, verbose=1, n_steps=32, nminibatches=4, tensorboard_log="./logs/")
	model.learn(total_timesteps=1000000)
	model.save("ppo2_model")

	del model # remove to demonstrate saving and loading

def test():
	make_env_def()
	# multiprocess environment
	env = make_vec_env('AI4U-v0', n_envs=8)
	
	model = PPO2.load("ppo2_model_baked", policy=CustomPolicy, tensorboard_log="./logs/")
	model.set_env(env)
	
	# Enjoy trained agent
	obs = env.reset()
	while True:
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		#env.render()

if __name__ == '__main__':
	test()

