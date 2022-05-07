import gym
import bandit_algorithms as alg
import environment as env
import victims
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env


class AttackModel(ABC):
    """Custom attacker model for testing hand designed attack policies"""
    @property
    @abstractmethod
    def victim(self):
        pass

    @abstractmethod
    def predict(self, obs):
        pass


class ZeroAttack(AttackModel):
    def __init__(self, victim: victims.BanditVictim):
        self.victim = victim

    def victim(self):
        return self.victim

    def predict(self, obs):
        return np.zeros(self.victim.action_space.shape), 0


class RandomAttack(AttackModel):
    def __init__(self, victim: victims.BanditVictim):
        self.victim = victim

    def victim(self):
        return self.victim

    def predict(self, obs):
        return np.zeros(self.victim.action_space.shape), 0


def test_attacker(model, victim):
    np.random.seed(1)

    num_iter = 10000
    num_tests = 10

    obs = victim.reset()
    avg_reward = np.zeros(num_iter)
    attacker_reward = np.zeros(num_iter)
    avg_cum_regret = np.zeros(num_iter)
    for k in tqdm(range(num_tests), position=1, colour='red'):
        cum_regret = 0
        victim.reset()

        for t in tqdm(range(num_iter), position=2, colour='blue', leave=False):
            action, _state = model.predict(obs)
            obs, reward, done, info = victim.step(action)

            victim_reward = info['victim_reward']
            victim_regret = info['victim_regret']

            attacker_reward[t] += 1 / (k + 1) * (reward - attacker_reward[t])
            avg_reward[t] += 1 / (k + 1) * (victim_reward - avg_reward[t])
            cum_regret += victim_regret
            avg_cum_regret[t] += 1 / (k + 1) * (cum_regret - avg_cum_regret[t])

    return avg_reward, avg_cum_regret, attacker_reward



def main():
    environment = env.generate_10_arm_testbed('StandardNormal')
    bandit = alg.Exp3Algorithm(10, 0.001)

    victim = victims.BanditVictim(bandit,
                                environment,
                                victims.ObservationModel.ACTION_REWARD,
                                victims.AttackModel.CHOSEN_ARM,
                                np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))


    ## No attack
    vanilla_model = ZeroAttack(victim)
    vanilla_victim_reward, vanilla_victim_cum_regret, vanilla_attacker_reward = test_attacker(vanilla_model, victim)

    plt.plot(vanilla_victim_cum_regret)
    plt.show()
    plt.plot(vanilla_attacker_reward)
    plt.show()

    # PPO_model = PPO('MlpPolicy', victim, verbose=1, n_epochs=1000)
    # PPO_model.learn(total_timesteps=10000)
    #
    # PPO_model.save("models/ppo_all_arms_policy")
    #
    # PPO_model = PPO.load("models/ppo_all_arms_policy")

    # SAC_model = SAC.load("models/sac_all_arms_action_reward3", victim)
    #
    # # SAC_model = SAC('MlpPolicy', victim,
    # #                 verbose=1,
    # #                 ent_coef='auto_0.1')
    # SAC_model.learn(total_timesteps=100000, log_interval=4)
    #
    # SAC_model.save("models/sac_all_arms_action_reward3")
    #
    # sac_victim_reward, sac_victim_cum_regret, sac_attacker_reward = test_attacker(SAC_model, victim)
    #
    # plt.plot(sac_victim_cum_regret)
    # plt.show()
    # plt.plot(sac_attacker_reward)
    # plt.show()

    # A2C_model = A2C('MlpPolicy',
    #                 victim,
    #                 verbose=0,
    #                 gamma=0.99)
    #
    # A2C_model.learn(total_timesteps=1000000,
    #                 log_interval=100)
    #
    # A2C_model.save("models/a2c_all_arms_action_reward3")
    #
    # A2C_model = A2C.load("models/a2c_all_arms_action_reward3")
    #
    # a2c_victim_reward, a2c_victim_cum_regret, a2c_attacker_reward = test_attacker(A2C_model, victim)
    #
    # plt.plot(a2c_victim_cum_regret)
    # plt.show()
    # plt.plot(a2c_attacker_reward)
    # plt.show()

    # DDPG_model = DDPG('MlpPolicy',
    #                 victim,
    #                 verbose=0,
    #                 gamma=0.99)
    #
    # DDPG_model.learn(total_timesteps=100000,
    #                 log_interval=100)
    #
    # DDPG_model.save("models/ddpg_all_arms_action_reward3")

    DDPG_model = DDPG.load("models/ddpg_all_arms_action_reward3")

    ddpg_victim_reward, ddpg_victim_cum_regret, ddpg_attacker_reward = test_attacker(DDPG_model, victim)

    plt.plot(ddpg_victim_cum_regret)
    plt.show()
    plt.plot(ddpg_attacker_reward)
    plt.show()



if __name__ == '__main__':
    main()