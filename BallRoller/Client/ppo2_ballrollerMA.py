import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from ai4u.utils import environment_definitions
import AI4UGym
from AI4UGym import BasicAgent
import numpy as np

def get_state_from_fields(fields):
	return np.array([fields['tx'], fields['tz'], fields['vx'], fields['vz'], fields['x'],  fields['z']])

class Agent(BasicAgent):
	def __init__(self):
		BasicAgent.__init__(self)


	def reset(self, env):
		env_info = env.remoteenv.step("restart")
		return get_state_from_fields(env_info)

	def act(self, env, action, info=None):
		for _ in range(4):
			envinfo = env.one_step(action)
			if envinfo['done']:
				break
		state = get_state_from_fields(envinfo)
		return state, envinfo['reward'], envinfo['done'], envinfo

def make_env_def():
	environment_definitions['state_shape'] = (6,)
	environment_definitions['action_shape'] = (5,)
	environment_definitions['actions'] = [('fx', 0.1), ('fx', -0.1), ('fz', 0.1), ('fz', -0.1), ('noop', 0.0)]
	environment_definitions['agent'] = Agent
	environment_definitions['input_port'] = 8080
	environment_definitions['output_port'] = 7070
	environment_definitions['host'] = '127.0.0.1'
	BasicAgent.environment_definitions = environment_definitions


make_env_def()

# multiprocess environment
env = make_vec_env('AI4U-v0', n_envs=4)

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=125000)
model.save("ppo2_ballroller")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_ballroller")

# Enjoy trained agent
obs = env.reset()
while True:
	action, _states = model.predict(obs)
	obs, rewards, dones, info = env.step(action)
	#env.render()
