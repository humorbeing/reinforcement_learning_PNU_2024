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
         random_state=42, **task_kwargs):
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
    #   time_limit=1
      )

from dm_control import viewer
if __name__ == '__main__':
    
    # env = _env(_home_team(1) + _away_team(0))
    env = _env()
    action_spec = env.action_spec()[0]
    # def random_policy(time_step):
    #     score_home = time_step[3][0]['stats_home_score']
    #     score_away = time_step[3][0]['stats_away_score']
    #     print(score_home, score_away, env.task.get_reward(None))
    #     return np.random.uniform(low=action_spec.minimum,
    #                             high=action_spec.maximum,
    #                             size=action_spec.shape)
    # viewer.launch(env, policy=random_policy)
    counter = 0
    _, reward, done, time_step = env.reset()
    for i in range(10000):
        # env.render()
        
        action = np.random.uniform(low=action_spec.minimum,
                                high=action_spec.maximum,
                                size=action_spec.shape)
        _, reward, done, time_step = env.step(action)
        score_home = time_step[0]['stats_home_score']
        score_away = time_step[0]['stats_away_score']
        print(score_home, score_away, env.task.get_reward(None))
        counter += 1
        if env.task.get_reward(None)[0]==0:
           done = False
        else:
           done = True

        if done:
            print('end')
            time_step = env.reset()    
        if counter > 500:
            print('limit')
            time_step = env.reset()
            counter = 0
print('end')