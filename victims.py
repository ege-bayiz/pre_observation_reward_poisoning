import gym
import numpy as np
from scipy.special import rel_entr
from gym import spaces
import bandit_algorithms as alg
import environment as env
from enum import Enum
import matplotlib.pyplot as plt
from tqdm import tqdm

LAMBDA = 1
BUDGET = 10
EPISODE_LENGTH = 10000
BETA = 0.1
class ObservationModel(Enum):
    """Defines the observations made by the attacker."""
    POLICY = 0
    ACTION_REWARD = 1


class AttackModel(Enum):
    """Defines the actions of the attacker."""
    CHOSEN_ARM = 0
    ALL_ARMS = 1


class BanditVictim(gym.Env):
    """A victim bandit algorithm implementation.
    Equivalently can be thought as the reward poisoning attacker's environment."""

    def __init__(self,
                 victim_alg: alg.BanditAlgorithm,
                 victim_env: env.Environment,
                 observation_model: ObservationModel,
                 attack_model: AttackModel,
                 target_policy: np.ndarray,
                 ):
        super(BanditVictim, self).__init__()
        self.alg = victim_alg
        self.env = victim_env
        self.obs_model = observation_model
        self.att_model = attack_model
        self.time = 0
        self.target_policy = target_policy

        if self.att_model == AttackModel.CHOSEN_ARM:
            self.action_space = spaces.Box(low=-BUDGET, high=BUDGET, shape=(1,))
        elif self.att_model == AttackModel.ALL_ARMS:
            self.action_space = spaces.Box(low=-BUDGET, high=BUDGET, shape=(self.env.num_arms,))
        else:
            raise ValueError(
                "Attack model not implemented."
            )

        if self.obs_model == ObservationModel.ACTION_REWARD:
            l_bound = -np.ones(self.env.num_arms + 2) #Actions, reward, time
            u_bound = np.ones(self.env.num_arms + 2)  #Actions, reward, time
            l_bound[-2] = -np.inf
            u_bound[-2] = np.inf
            l_bound[-1] = -np.inf
            u_bound[-1] = np.inf
            self.observation_space = spaces.Box(low=l_bound, high=u_bound)
        elif self.obs_model == ObservationModel.POLICY:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.env.num_arms,))
        else:
            raise ValueError(
                "Observation model not implemented."
            )

    def reset(self):
        # Reset the state of the environment to an initial state
        self.time = 0
        self.alg.reset()

        if self.obs_model == ObservationModel.POLICY:
            obs = self.alg.get_arm_probs()
        elif self.obs_model == ObservationModel.ACTION_REWARD:
            obs = np.zeros(self.env.num_arms + 2)
        else:
            raise ValueError(
                "Observation model not implemented."
            )

        return obs

    def step(self, action):
        # Execute one time step within the environment
        self.time += 1
        if self.att_model == AttackModel.CHOSEN_ARM:
            arm = self.alg.choose_arm()
            vict_reward, _ = self.env.pull_arm(arm)
            poisoned_reward = vict_reward + action  # Adding the "poison"
            self.alg.update(arm=arm, reward=poisoned_reward)
            attack_cost = np.abs(action[0])

            #computing victim regret (For diagnostic purposes)
            if arm == self.env.best_arm:
                vict_regret = 0
            else:
                vict_regret, _ = self.env.pull_arm(self.env.best_arm)
                vict_regret -= vict_reward

        elif self.att_model == AttackModel.ALL_ARMS:
            arm = self.alg.choose_arm()
            vict_reward, _ = self.env.pull_arm(arm)
            poisoned_reward = vict_reward + action[arm]  # Adding the "poison"
            self.alg.update(arm=arm, reward=poisoned_reward)
            attack_cost = np.linalg.norm(action)

            #computing victim regret (For diagnostic purposes)
            if arm == self.env.best_arm:
                vict_regret = 0
            else:
                vict_regret, _ = self.env.pull_arm(self.env.best_arm)
                vict_regret -= vict_reward

        else:
            raise ValueError(
                "Attack model not implemented."
            )

        if self.obs_model == ObservationModel.POLICY:
            obs = self.alg.get_arm_probs()
            obs_cost = np.linalg.norm(obs - self.target_policy, 1)
        elif self.obs_model == ObservationModel.ACTION_REWARD:
            arms = np.zeros(self.env.num_arms)
            arms[arm] = 1
            obs = np.zeros(self.env.num_arms + 2)
            obs[0:-2] = arms
            obs[-2] = vict_reward
            obs[-1] = self.time
            obs_cost = np.linalg.norm(arms - self.target_policy, 1)
        else:
            raise ValueError(
                "Observation model not implemented."
            )

        attacker_reward = 1 / (obs_cost + BETA) - LAMBDA * attack_cost
        done = (self.time == EPISODE_LENGTH)

        return obs, attacker_reward, done, {'victim_reward': vict_reward,
                                            'victim_regret': vict_regret}

    def render(self, mode='human', close=False):
        print(self.alg.get_arm_probs())



def test_bandit_victim(vict: BanditVictim, num_iter: int, num_tests: int):
    avg_reward = np.zeros(num_iter)
    avg_cum_regret = np.zeros(num_iter)
    for k in tqdm(range(num_tests), position=1, colour='red'):
        cum_regret = 0
        vict.reset()
        for t in tqdm(range(num_iter), position=2, colour='blue', leave=False):
            obs, attacker_reward, done, info = vict.step(0)

            victim_reward = info['victim_reward']
            victim_regret = info['victim_regret']

            avg_reward[t] += 1 / (k + 1) * (victim_reward - avg_reward[t])

            cum_regret += victim_regret
            avg_cum_regret[t] += 1 / (k + 1) * (cum_regret - avg_cum_regret[t])

    plt.plot(avg_cum_regret)
    plt.show()

# environment = env.generate_10_arm_testbed('StandardNormal')
# bandit = alg.Exp3Algorithm(10, 0.01)
# vict = BanditVictim(environment,
#                         bandit,
#                         ObservationModel.POLICY,
#                         AttackModel.CHOSEN_ARM,
#                         )
#
# test_bandit_victim(vict, 100000, 10)