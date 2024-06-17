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

    def action_spec(self):
        return self.action_space


# check_env(env)
env = make_vec_env(CustomEnv)
# env = CustomEnv()

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

# ppopath = '/home/ray/workspace/codes/playground/reinforcement_learning_PNU_2024/homeworks/project01/dataset/transfer/go_to_target/general_low_level/best_model.zip'
ppopath = '/home/ray/workspace/codes/playground/reinforcement_learning_PNU_2024/homeworks/project01/playground_mocapaction/testing/5/me.zip'
modelppo = PPO.load(ppopath)
from dm_control import viewer

def random_policy(time_step):
    obs = get_obs(time_step[3])
    with torch.no_grad():
        embed = modelppo.predict(obs, deterministic=True)
        temp11 = torch.tensor(embed[0])[None,...]
        temp12 = distilled_model.low_level_policy(obs, temp11)
    
    action = temp12.numpy()[0]
    return action
    # print('end')



# env.action_spec = ENV.action_spec
# env.physics = ENV.physics
viewer.launch(ENV, policy=random_policy)







print('end')