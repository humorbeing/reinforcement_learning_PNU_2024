import unittest

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion import soccer
from dm_control.locomotion.soccer import camera
from dm_control.locomotion.soccer import initializers
from dm_control.mujoco.wrapper import mjbindings
import numpy as np

RGBA_BLUE = [.1, .1, .8, 1.]
RGBA_RED = [.8, .1, .1, 1.]
from dm_control.locomotion.walkers.initializers import mocap
from dm_control.locomotion.soccer.humanoid import Humanoid
initializer = mocap.CMUMocapInitializer()

def _walker(name, walker_id, marker_rgba):
  return soccer.Humanoid(
    name=name,
    walker_id=walker_id,
    marker_rgba=marker_rgba,
    visual=Humanoid.Visual.JERSEY,
    initializer=initializer
  )


def _team_players(team_size, team, team_name, team_color):
  team_of_players = []
  for i in range(team_size):
    team_of_players.append(
        soccer.Player(team, _walker("%s%d" % (team_name, i), i, team_color)))
  return team_of_players


def _home_team(team_size):
  return _team_players(team_size, soccer.Team.HOME, "home", RGBA_BLUE)


def _away_team(team_size):
  return _team_players(team_size, soccer.Team.AWAY, "away", RGBA_RED)

from dm_control.locomotion.soccer.soccer_ball import SoccerBall

ball = SoccerBall(
    radius=0.117,
    mass=0.45,
    friction=(0.7,0.05,0.04),
    damp_ratio=0.4)

def _env(players=_home_team(1), disable_walker_contacts=True, observables=None,
         random_state=42, time_limit=float('inf'), **task_kwargs):
    return composer.Environment(
        task=soccer.Task(
            players=players,
            arena=soccer.Pitch((3, 3), goal_size=(0.61, 1.83, 0.61)),
            ball=ball,
            observables=observables,
            disable_walker_contacts=disable_walker_contacts,
            **task_kwargs
        ),
        random_state=random_state,
        time_limit=time_limit
        )

def _env1(players=_home_team(1), disable_walker_contacts=True, observables=None,
         random_state=42, time_limit=float('inf'), **task_kwargs):
#   time_limit=float('inf')
  return composer.Environment(
      task=soccer.Task(
          players=players,
          arena=soccer.Pitch((3, 3), goal_size=(0.61, 1.83, 0.61)),
          ball=ball,
          observables=observables,
          disable_walker_contacts=disable_walker_contacts,
          **task_kwargs
      ),
      random_state=random_state,
      time_limit=time_limit
      )
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import gym
from gym import spaces
ENV = _env()

def get_obs(obs):
    ob_list = []
    keysss = ['appendages_pos', 'body_height', 'end_effectors_pos', 'joints_pos', 'joints_vel', 'prev_action', 'sensors_accelerometer', 'sensors_gyro', 'sensors_velocimeter', 'world_zaxis', 'ball_ego_angular_velocity', 'ball_ego_position', 'ball_ego_linear_velocity', 'team_goal_back_right', 'team_goal_mid', 'team_goal_front_left', 'field_front_left', 'opponent_goal_back_left', 'opponent_goal_mid', 'opponent_goal_front_right', 'field_back_right', 'stats_vel_to_ball', 'stats_closest_vel_to_ball', 'stats_veloc_forward', 'stats_vel_ball_to_goal', 'stats_home_avg_teammate_dist', 'stats_home_score', 'stats_away_score']
    # keysss = ['joints_pos', 'joints_vel', 'sensors_velocimeter', 'sensors_gyro', 'end_effectors_pos', 'world_zaxis', 'appendages_pos', 'body_height', , , , 'prev_action', 'sensors_accelerometer', , , , 'ball_ego_angular_velocity', 'ball_ego_position', 'ball_ego_linear_velocity', 'team_goal_back_right', 'team_goal_mid', 'team_goal_front_left', 'field_front_left', 'opponent_goal_back_left', 'opponent_goal_mid', 'opponent_goal_front_right', 'field_back_right', 'stats_vel_to_ball', 'stats_closest_vel_to_ball', 'stats_veloc_forward', 'stats_vel_ball_to_goal', 'stats_home_avg_teammate_dist', 'stats_home_score', 'stats_away_score']
    for key in keysss:
        temp11 = obs[0][key]
        if len(temp11.shape) == 2:
            temp11 = temp11[0]
        # print(f'{key}-{type(temp11)}{temp11.shape}-{temp11}')
        for i in temp11:
            # print(i)        
            ob_list.append(i)
    return np.array(ob_list)
    # print('s')


# obs = ENV.reset()

# get_obs(obs[3])
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.env = ENV
        self.action_space = spaces.Box(high=1, low=-1,shape=(56,), dtype=np.float64)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(high=np.inf, low=-np.inf,shape=(242,), dtype=np.float64)

    def step(self, action):
        _, r, _, obs = self.env.step(action)
        reward = float(r[0])
        obser = get_obs(obs)
        if reward == 0:
           done = False
        else:
           done = True
        return obser, reward, done, {}

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return get_obs(obs[3])

    def render(self):
        pass

    def close(self):
        self.env.close()

# ENV1 = _env1(time_limit=10)
# class CustomEnv1(gym.Env):
#     """Custom Environment that follows gym interface."""

    # metadata = {"render_modes": ["human"], "render_fps": 30}

    # def __init__(self,):
    #     super().__init__()
    #     # Define action and observation space
    #     # They must be gym.spaces objects
    #     # Example when using discrete actions:
    #     self.env = ENV1
    #     self.action_space = spaces.Box(high=1, low=-1,shape=(56,), dtype=np.float64)
    #     # Example for using image as input (channel-first; channel-last also works):
    #     self.observation_space = spaces.Box(high=np.inf, low=-np.inf,shape=(242,), dtype=np.float64)

    # def step(self, action):
    #     _, r, _, obs = self.env.step(action)
    #     reward = float(r[0])
    #     obser = get_obs(obs)
    #     if reward == 0:
    #        done = False
    #     else:
    #        done = True
    #     return obser, reward, done, {}

    # def reset(self, seed=None, options=None):
    #     obs = self.env.reset()
    #     return get_obs(obs[3])

    # def render(self):
    #     pass

    # def close(self):
    #     self.env.close()


# check_env(env)
env = make_vec_env(CustomEnv)

import model
import wrappers

low_level_policy_path = '/home/ray/workspace/codes/playground/reinforcement_learning_PNU_2024/homeworks/project01/playground_mocapaction/haha/model/low_level_policy.ckpt'
distilled_model = model.NpmpPolicy.load_from_checkpoint(
    low_level_policy_path,
    map_location='cpu'
)
env = wrappers.EmbedToActionVecWrapper(
    env,
    distilled_model.embed_size,
    max_embed=3.0,
    embed_to_action=distilled_model.low_level_policy
)

# env = DummyVecEnv([env])
# env = DummyVecEnv([lambda: env] )
from stable_baselines3 import PPO

import torch

from mocapact.sb3 import callbacks
from stable_baselines3.common.callbacks import EvalCallback

eval_env = CustomEnv()
eval_freq = 100000
eval_model_path = 'haha/model'
eval_path = 'haha/eval'
callback_on_new_best = callbacks.SaveVecNormalizeCallback(
    save_freq=1,
    save_path=eval_model_path
)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=eval_model_path,
    log_path=eval_path,
    eval_freq=eval_freq,
    callback_on_new_best=callback_on_new_best,
    # callback_after_eval=callbacks.SeedEnvCallback(FLAGS.eval.seed),
    n_eval_episodes=10,
    # deterministic=True,
    )


layer_sizes = 3 * [1024]

policy_kwargs = dict(
    net_arch=[dict(pi=layer_sizes, vf=layer_sizes)],
    activation_fn=torch.nn.modules.activation.Tanh,
    log_std_init=np.log(2.5),
)

batch_size = 10000
n_steps = 10000
target_kl = 0.15
learning_rate = 0.0001
max_grad_norm = 1.0
model = PPO("MlpPolicy", env,
            batch_size=batch_size,
            n_steps=n_steps,
            target_kl=target_kl,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log='haha/tb',
            verbose=1)

callback = [
        eval_callback,
    ]
model.learn(10000000)
model.save('me')
# model.learn(1000000, callback=callback)


print('end')