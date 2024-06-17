import gym
import dm_control_wrapper
import env_util

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import wrappers

from dm_control.locomotion.tasks import go_to_target

task_ = go_to_target.GoToTarget
n_workers = 1
max_episode_steps = 999

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

# env = make_env()  # strange error
if __name__ == '__main__':
    
    env = make_env()  # strange running order
    

    model = PPO("MultiInputPolicy", env)
    model.learn(1)


    env_ctor = dm_control_wrapper.DmControlWrapper.make_env_constructor(task_)
    task_kwargs = dict(
        physics_timestep=0.005,
        control_timestep=0.03,
        moving_target=True
    )
    environment_kwargs = dict(
        time_limit=24.99,
        random_state=0
    )
    big_arena = False
    arena_size = (1000., 1000.) if big_arena else (8., 8.)
    env = env_ctor(
        task_kwargs=task_kwargs,
        environment_kwargs=environment_kwargs,
        arena_size=arena_size
    )

    import torch
    @torch.no_grad()
    def policy_fn(time_step):
        obs = env.get_observation(time_step)
        action = model.predict(obs, deterministic=True)
         
        return action[0]
    
    from dm_control.viewer import application
    viewer_app = application.Application(title='Humanoid Task', width=1024, height=768)
    viewer_app.launch(environment_loader=env.dm_env, policy=policy_fn) 

print('end')