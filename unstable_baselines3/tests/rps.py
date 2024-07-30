import numpy as np

from pettingzoo.classic import rps_v2

from stable_baselines3.dqn.policies import MlpPolicy as DQNPolicy
from stable_baselines3.dqn.dqn import DQN

from stable_baselines3.ppo.policies import MlpPolicy as PPOPolicy

from unstable_baselines3 import WorkerPPO, WorkerDQN
from unstable_baselines3.common.multi_agent_alg import DumEnv
from unstable_baselines3.common.auto_multi_alg import AutoMultiAgentAlgorithm
from stable_baselines3.common.utils import spaces
import os, sys

Worker = WorkerDQN

kwargs = {'batch_size': 128,
          'gamma': 0,
          }
if issubclass(Worker, DQN):
    MlpPolicy = DQNPolicy
    kwargs['learning_starts'] = 128
else:
    MlpPolicy = PPOPolicy
    kwargs['n_steps'] = 128

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

# both work
env = rps_v2.env()
#env = rps_v2.parallel_env()


env.reset()

action_space = env.action_space('player_0')
obs_space = env.observation_space('player_0')


class always_0:
    def get_action(self, *args, **kwargs):
        return 0


class easy_pred:
    def __init__(self, p=.01):
        self.p = p

    def get_action(self, obs, *args, **kwargs):
        # if initialized game or random chance
        if obs == 3 or np.random.random() < self.p:
            self.choice = np.random.randint(3)

        return self.choice


# we will first set player 1 to usually repeat moves, and train a PPO agent to beat it

worker0 = Worker(env=DumEnv(action_space=action_space,
                            obs_space=obs_space,
                            ),
                 policy=MlpPolicy,
                 **kwargs,
                 )

thingy = AutoMultiAgentAlgorithm(policy=MlpPolicy,
                                 env=env,
                                 worker_infos={
                                     'player_0': {'train': True},
                                     # default value is True, so including this does nothing
                                     'player_1': {'train': False},
                                 },
                                 workers={'player_0': worker0,
                                          'player_1': easy_pred(),
                                          },

                                 # DefaultWorkerClass=Worker,
                                 # DefaultWorkerClass is only for if all players are not specified in workers
                                 )
print('starting training1')
thingy.learn(total_timesteps=4096*4)
print('trained')

# display
disp_env = rps_v2.parallel_env(render_mode="human")

thingy = AutoMultiAgentAlgorithm(policy=MlpPolicy,
                                 env=disp_env,
                                 worker_infos={
                                     'player_0': {'train': False},
                                     'player_1': {'train': False},
                                 },
                                 workers={'player_0': worker0,
                                          'player_1': easy_pred()},

                                 )

thingy.learn(total_timesteps=10)
