import numpy as np

from pettingzoo.classic import rps_v2

from stable_baselines3.dqn.policies import MlpPolicy as DQNPolicy
from stable_baselines3.dqn.dqn import DQN

from stable_baselines3.ppo.policies import MlpPolicy as PPOPolicy

from unstable_baselines3.ppo.PPO import WorkerPPO
from unstable_baselines3.common.multi_agent_alg import DumEnv
from unstable_baselines3.common.auto_multi_alg import AutoMultiAgentAlgorithm
from stable_baselines3.common.utils import spaces
import os, sys

Worker = WorkerPPO

if issubclass(Worker, DQN):
    MlpPolicy = DQNPolicy
else:
    MlpPolicy = PPOPolicy

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

# env = rps_v2.env()
env = rps_v2.parallel_env()
env.reset()

action_space = env.action_space('player_0')
obs_space = env.observation_space('player_0')


class always_0:
    def get_action(self, *args, **kwargs):
        return 0


class easy_pred:
    def __init__(self, p=.01):
        self.choice = 0
        self.p = p

    def get_action(self, obs, *args, **kwargs):
        # if initialized game or random chance
        if obs == 4 or np.random.random() < self.p:
            self.choice = np.random.randint(3)

        return self.choice


thingy = AutoMultiAgentAlgorithm(policy=MlpPolicy,
                                 env=env,
                                 DefaultWorkerClass=Worker,
                                 worker_infos={  # 'player_0': {'train': True},
                                     # default value is True, so including this does nothing
                                     'player_1': {'train': False},
                                 },
                                 workers={'player_0': Worker(env=DumEnv(action_space=action_space,
                                                                        obs_space=obs_space,
                                                                        ),
                                                             policy=MlpPolicy,
                                                             n_steps=100,
                                                             batch_size=100,
                                                             gamma=0.,
                                                             ),
                                          'player_1': easy_pred(),
                                          },

                                 )
print('starting training')
thingy.learn(total_timesteps=5000)
print('trained')

worker0 = thingy.workers['player_0']
worker0: Worker

env = rps_v2.parallel_env(render_mode="human")

thingy = AutoMultiAgentAlgorithm(policy=MlpPolicy,
                                 env=env,
                                 # DefaultWorkerClass=Worker,
                                 worker_infos={
                                     'player_0': {'train': False},
                                     'player_1': {'train': False},
                                 },
                                 workers={'player_0': worker0,
                                          'player_1': easy_pred()},

                                 )

thingy.learn(total_timesteps=10)
