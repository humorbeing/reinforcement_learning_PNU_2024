from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.tasks import go_to_target

from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.mocap import loader
from dm_control.locomotion.walkers import initializers
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.reference_pose import utils
import tree
import mujoco

class StandInitializer(initializers.WalkerInitializer):
    def __init__(self):
        ref_path = cmu_mocap_data.get_path_for_cmu(version='2020')
        mocap_loader = loader.HDF5TrajectoryLoader(ref_path)
        trajectory = mocap_loader.get_trajectory('CMU_040_12')
        clip_reference_features = trajectory.as_dict()
        clip_reference_features = tracking._strip_reference_prefix(clip_reference_features, 'walker/')
        self._stand_features = tree.map_structure(lambda x: x[0], clip_reference_features)

    def initialize_pose(self, physics, walker, random_state):
        del random_state
        utils.set_walker_from_features(physics, walker, self._stand_features)
        mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)



initializer = StandInitializer()
walker = cmu_humanoid.CMUHumanoidPositionControlledV2020(initializer=initializer)
# walker = cmu_humanoid.CMUHumanoidPositionControlledV2020()
arena = floors.Floor((8.0, 8.0))
task_kwargs = {'physics_timestep': 0.005, 'control_timestep': 0.03, 'moving_target': True}
task = go_to_target.GoToTarget(
    walker,
    arena,
    **task_kwargs
)

from dm_control import composer
import numpy as np
environment_kwargs = {}
rng = np.random.RandomState(seed=1)
environment_kwargs['random_state'] = rng
env = composer.Environment(
    task=task,
    **environment_kwargs
)
task.random = env.random_state
_original_rng_state = env.random_state.get_state()

# Set observation and actions spaces
# _observation_space = self._create_observation_space()
from gym import spaces
obs_spaces = dict()
for k, v in env.observation_spec().items():
    if v.dtype == np.float64 and np.prod(v.shape) > 0:
        if np.prod(v.shape) > 0:
            obs_spaces[k] = spaces.Box(
                -np.infty,
                np.infty,
                shape=(np.prod(v.shape),),
                dtype=np.float32
            )
    elif v.dtype == np.uint8:
        tmp = v.generate_value()
        obs_spaces[k] = spaces.Box(
            v.minimum.item(),
            v.maximum.item(),
            shape=tmp.shape,
            dtype=np.uint8
        )
_observation_space = spaces.Dict(obs_spaces)

action_spec = env.action_spec()
dtype = np.float32

_action_space = spaces.Box(
    low=action_spec.minimum.astype(dtype),
    high=action_spec.maximum.astype(dtype),
    shape=action_spec.shape,
    dtype=dtype
)
wrapper_kwargs = {'max_episode_steps': 833}
import gym
env.spec = None
env = gym.wrappers.TimeLimit(
    env,
    **wrapper_kwargs
)

# from stable_baselines3.common.vec_env import VecNormalize
# env = VecNormalize(env, training=True, gamma=0.99,
#     norm_obs=True,
#     norm_reward=True)
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
    # if FLAGS.low_level_policy_path:
    #     distilled_model = model.NpmpPolicy.load_from_checkpoint(
    #         FLAGS.low_level_policy_path,
    #         map_location='cpu'
    #     )
    #     env = wrappers.EmbedToActionVecWrapper(
    #         env,
    #         distilled_model.embed_size,
    #         max_embed=FLAGS.max_embed,
    #         embed_to_action=distilled_model.low_level_policy
    #     )
    env = VecNormalize(env, training=training, gamma=0.99,
                       norm_obs=True,
                       norm_reward=True)
    return env

env = make_env()

from stable_baselines3 import PPO

model = PPO("MultiInputPolicy", env)


model.learn(1)

print('end')