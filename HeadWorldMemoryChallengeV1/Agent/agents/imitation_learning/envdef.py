import AI4UGym
from AI4UGym import BasicAgent
from ai4u.utils import environment_definitions
import numpy as np
from ai4u.utils import image_decode

TOUCH_SIZE = 8
IMAGE_SHAPE = (21, 20, 1)
ARRAY_SIZE = 20
ACTION_SIZE = 5


def get_frame_from_fields(fields, touchs):
	imgdata = image_decode(fields['frame'], 20, 20, np.uint8)
	proprioceptions = np.zeros(ARRAY_SIZE)
	for i in range(TOUCH_SIZE):
		proprioceptions[i] = touchs[i]
	proprioception = np.array(proprioceptions, dtype=np.uint8)
	imgdata = np.vstack([imgdata, proprioception])
	return imgdata

class HumanAgent(BasicAgent):
	def __init__(self):
		BasicAgent.__init__(self)

	def __make_state__(imageseq):
		frameseq = np.array([imageseq], dtype=np.uint8)
		frameseq = np.moveaxis(frameseq, 0, -1)
		return frameseq

	def reset(self, env):
		env_info = env.remoteenv.step("restart")
		return HumanAgent.__make_state__( get_frame_from_fields(env_info, np.zeros(8)) )

	def act(self, env, action, info=None):
		touch_sensor = np.zeros(TOUCH_SIZE)
		sum_reward = 0
		for f in range(TOUCH_SIZE):
			envinfo = env.one_step(action)
			touch_sensor[f] = envinfo['touched']
			sum_reward += envinfo['reward']
			if envinfo['done']:
				break
		frame = get_frame_from_fields(envinfo, touch_sensor)
		state = HumanAgent.__make_state__(frame)
		return state, sum_reward, envinfo['done'], envinfo

def make_env_def(agent):
	environment_definitions['state_shape'] = IMAGE_SHAPE
	environment_definitions['action_shape'] = (ACTION_SIZE,)
	environment_definitions['state_type'] = np.uint8
	environment_definitions['actions'] = [('act', 0), ('act', 1), ('act', 3), ('act', 4), ('act', -1)]
	environment_definitions['agent'] = agent
	BasicAgent.environment_definitions = environment_definitions