import metaworld
import gym
from gym.envs.registration import register
from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS, ALL_V2_ENVIRONMENTS

gym.logger.set_level(40)
print(metaworld.MT1.ENV_NAMES)


def assert_env(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert hasattr(env, '_max_episode_steps')


for task in metaworld.MT1.ENV_NAMES:
    register(
        id=task,
        entry_point=ALL_V1_ENVIRONMENTS[task],
        max_episode_steps=150)
    assert_env(gym.make(task))


def make_env(env_id, task=None):
    env = gym.make(env_id)
    setattr(env, 'is_metaworld', env_id in metaworld.MT1.ENV_NAMES)
    if task is not None:
        env.set_task(task)
    return env
