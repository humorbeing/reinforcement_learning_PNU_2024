import gym
import dm_control_wrapper
import env_util
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import wrappers

from dm_control.locomotion.tasks import go_to_target

task_ = go_to_target.GoToTarget
n_workers = 1
max_episode_steps = 833

def make_env(seed=0, training=True):
    env_id = dm_control_wrapper.DmControlWrapper.make_env_constructor(task_)
    task_kwargs = dict(
        physics_timestep=0.005,
        control_timestep=0.03,
        moving_target=True
    )
    env_kwargs = dict(task_kwargs=task_kwargs)
    
    env = env_util.make_vec_env(
        env_id=env_id,
        n_envs=n_workers,
        seed=seed,
        wrapper_class=gym.wrappers.TimeLimit,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        vec_monitor_cls=wrappers.VecMonitor,
        wrapper_kwargs=dict(max_episode_steps=max_episode_steps)
    )    
    env = VecNormalize(env, training=training, gamma=0.99,
                       norm_obs=True,
                       norm_reward=True)
    return env

# env = make_env()
if __name__ == '__main__':    
    env = make_env()
    print('start')

print('end')