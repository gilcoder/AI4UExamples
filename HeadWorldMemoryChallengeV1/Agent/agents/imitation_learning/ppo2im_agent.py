import gym

from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from ai4u.utils import environment_definitions
import AI4UGym
from AI4UGym import BasicAgent
import numpy as np
#from stable_baselines.common.tf_layers import conv, linear, conv_to_fc
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc
from stable_baselines.deepq.policies import FeedForwardPolicy
from ai4u.utils import image_decode
from collections import deque
from stable_baselines.gail import ExpertDatasetLSTM
import getch

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
	environment_definitions['state_type'] = np.uint8
	environment_definitions['actions'] = [('act', 0), ('act', 1), ('act', 3), ('act', 4), ('act', -1)]
	environment_definitions['agent'] = Agent
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
			env.remoteenv.step(np.random.choice([0, 3, 4]))
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
		
		if model is not None:
			prob = model.action_probability( np.array([state]*8) )[0]
			entropy = -np.sum(prob * np.log2(prob))
			if self.hist == 30 and sum_reward == 0:
				m = np.mean(hist)
				d = np.std(hist)
				if entropy >= m + d:
					sum_reward += 0.001
				else:
					sum_reward -= 0.001
			self.hist.append(entropy)
		return state, sum_reward, envinfo['done'], envinfo

def maker_pretrain_optimizer():
	import tensorflow as tf
	return tf.train.RMSPropOptimizer(learning_rate=0.00001, decay=0.9, momentum=0.0, epsilon=1e-10, 
		use_locking=False, centered=False, name='RMSProp')


def pretrain(bs=128, logdir='./logs/', expert_path="dummy_expert_ai4u2.npz", model_path="model", 
				traj_limitation=-1, n_envs=1, n_epochs = 100000, learning_rate=0.001, verbose=1, train_frac=0.9, val_interval=None):
	# Using only one expert trajectory
	# you can specify `traj_limitation=-1` for using the whole dataset
	dataset = ExpertDatasetLSTM(expert_path=expert_path,
							traj_limitation=traj_limitation, batch_size=bs, envs_per_batch=1, verbose=verbose, train_fraction=train_frac)
	make_env_def()
	env = make_vec_env('AI4U-v0', n_envs=n_envs)
	#batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
	model = PPO2(CustomPolicy, env, verbose=1, nminibatches=1, n_steps=bs, tensorboard_log=logdir)
	# Pretrain the PPO2 model
	model.pretrain(dataset, n_epochs=n_epochs, learning_rate=learning_rate, 
		val_interval=val_interval, maker_optimizer=maker_pretrain_optimizer)
	# As an option, you can train the RL agent
	# model.learn(int(1e5))
	model.save(model_path)
	del model # remove to demonstrate saving and loading


def train(dest_path="ppo2model", logdir='./logs/', pretrainedmodel=None, nsteps = 1, total_timesteps=10000, n_envs=1, verbose=1, nminibatches=4):
	model = None
	make_env_def()
	env = make_vec_env('AI4U-v0', n_envs=n_envs)
	if pretrainedmodel is not None:
		model = PPO2.load(pretrainedmodel, policy=CustomPolicy, tensorboard_log=logdir, nminibatches=nminibatches)
		model.set_env(env)
	else:
		model = PPO2(CustomPolicy,env, verbose=verbose, nminibatches=nminibatches, n_steps=nsteps, tensorboard_log=logdir)
	model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, tb_log_name=logdir)
	model.save(dest_path)

def test(model_path='model', n_envs=1, steps_by_episode=100000):
	# Test the pre-trained model
	# multiprocess environment
	make_env_def()
	env = make_vec_env('AI4U-v0', n_envs=n_envs)
	model = PPO2.load("model", env, policy=CustomPolicy)
	q = ' '
	while q != 's':
		obs = env.reset()
		reward_sum = 0.0
		for _ in range(steps_by_episode):
				action, _ = model.predict(obs)
				obs, reward, done, _ = env.step(action)
				reward_sum += reward
				env.render()
				if done:
						print(reward_sum)
						reward_sum = 0.0
						obs = env.reset()
		print('Exit?')
		q = getch.getche()
	env.close()


if __name__ == '__main__':
	train(n_envs=8, nminibatches=4, total_timesteps=1000000)
	#test()
	#pretrain(bs=521, n_envs=1, n_epochs=1000, learning_rate=0.00001, val_interval=100, train_frac=0.95)