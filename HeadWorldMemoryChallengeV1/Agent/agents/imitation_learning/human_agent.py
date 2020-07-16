import gym
import AI4UGym
from AI4UGym import BasicAgent
from stable_baselines.gail import generate_expert_traj
import getch
import envdef

envdef.make_env_def(envdef.HumanAgent)
actions = {'w':0, 's':2, 'd':1, 'a':3}
env = gym.make("AI4U-v0")
# Here the expert is a random agent
# but it can be any python function, e.g. a PID controller


def ai4u_expert(_obs):
    """
    Random agent. It samples actions randomly
    from the action space of the environment.

    :param _obs: (np.ndarray) Current observation
    :return: (np.ndarray) action taken by the expert
    """
    print(_obs.shape)
    print(_obs.dtype)
    action = getch.getche()

    return actions[action]
# Data will be saved in a numpy archive named `expert_cartpole.npz`
# when using something different than an RL expert,
# you must pass the environment object explicitly
generate_expert_traj(ai4u_expert, 'dummy_expert_ai4u2', env, image_folder="./images", n_episodes=10)