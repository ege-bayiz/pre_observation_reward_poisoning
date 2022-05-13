import gym
import bandit_algorithms as alg
import environment as env
import victims
import matplotlib.pyplot as plt
from tqdm import tqdm
from tempfile import TemporaryFile
import numpy as np
import os
import palette
from abc import ABC, abstractmethod
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env

NUM_SEEDS = 10
NUM_ITER = 10000
NUM_TESTS = 30


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
        np.random.seed(1)
        self.victim = victim

    def victim(self):
        return self.victim

    def predict(self, obs):
        return self.victim.action_space.sample(), 0


def test_attacker(model, victim, num_iter, num_tests):
    seed = 0
    np.random.seed(seed)
    model.set_random_seed(seed)

    victim_rewards = np.zeros((num_tests, num_iter))
    attacker_rewards = np.zeros((num_tests, num_iter))
    victim_regrets = np.zeros((num_tests, num_iter))
    arm_choices = np.zeros((num_tests, num_iter))
    for k in tqdm(range(num_tests), position=1, colour='red'):
        cum_regret = 0
        obs = victim.reset()

        for t in tqdm(range(num_iter), position=2, colour='blue', leave=False):
            action, _state = model.predict(obs)
            obs, attacker_reward, done, info = victim.step(action)

            victim_reward = info['victim_reward']
            victim_regret = info['victim_regret']
            arm_choice = info['arm_choice']

            victim_rewards[k, t] = victim_reward
            attacker_rewards[k, t] = attacker_reward
            victim_regrets[k, t] = victim_regret
            arm_choices[k, t] = arm_choice

    return victim_rewards, attacker_rewards, victim_regrets, arm_choices

    # seed = 3.1415
    # np.random.seed(seed)
    # model.set_random_seed(seed)
    # num_iter = 10000
    # num_tests = 30
    #
    # avg_reward = np.zeros(num_iter)
    # attacker_reward = np.zeros(num_iter)
    # avg_cum_regret = np.zeros(num_iter)
    # for k in tqdm(range(num_tests), position=1, colour='red'):
    #     cum_regret = 0
    #     obs = victim.reset()
    #
    #     for t in tqdm(range(num_iter), position=2, colour='blue', leave=False):
    #         action, _state = model.predict(obs)
    #         obs, reward, done, info = victim.step(action)
    #
    #         victim_reward = info['victim_reward']
    #         victim_regret = info['victim_regret']
    #
    #         attacker_reward[t] += 1 / (k + 1) * (reward - attacker_reward[t])
    #         avg_reward[t] += 1 / (k + 1) * (victim_reward - avg_reward[t])
    #         cum_regret += victim_regret
    #         avg_cum_regret[t] += 1 / (k + 1) * (cum_regret - avg_cum_regret[t])
    #
    # return avg_reward, avg_cum_regret, attacker_reward

def train_attackers(victim, victim_name, gamma):
    for seed in range(0, NUM_SEEDS):
        # PPO
        PPO_model = PPO('MlpPolicy',
                        victim,
                        verbose=1,
                        gamma=gamma,
                        seed=seed)
        PPO_model.learn(total_timesteps=100000)
        PPO_model.save("models/ppo/{victim_name}/{gamma}/ppo_{victim_name}_{gamma}_{seed}".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed))
        # PPO_model = PPO.load("models/ppo_chosen_arm_action_reward", victim)
        # ppo_victim_reward, ppo_victim_cum_regret, ppo_attacker_reward = test_attacker(PPO_model, victim)

        # SAC
        SAC_model = SAC('MlpPolicy',
                        victim,
                        verbose=1,
                        gamma=gamma,
                        seed=seed)
        SAC_model.learn(total_timesteps=100000)
        SAC_model.save("models/sac/{victim_name}/{gamma}/sac_{victim_name}_{gamma}_{seed}".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed))
        # SAC_model = SAC.load("models/sac_chosen_arm_action_reward", victim)
        # sac_victim_reward, sac_victim_cum_regret, sac_attacker_reward = test_attacker(SAC_model, victim)

        # A2C
        A2C_model = A2C('MlpPolicy',
                        victim,
                        verbose=1,
                        gamma=gamma,
                        seed=seed)
        A2C_model.learn(total_timesteps=100000)
        A2C_model.save("models/a2c/{victim_name}/{gamma}/a2c_{victim_name}_{gamma}_{seed}".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed))
        # A2C_model = A2C.load("models/a2c_chosen_arm_action_reward")
        # a2c_victim_reward, a2c_victim_cum_regret, a2c_attacker_reward = test_attacker(A2C_model, victim)

        # DDPG
        DDPG_model = DDPG('MlpPolicy',
                          victim,
                          verbose=1,
                          gamma=gamma,
                          seed=seed)
        DDPG_model.learn(total_timesteps=100000, log_interval=1)
        DDPG_model.save("models/ddpg/{victim_name}/{gamma}/ddpg_{victim_name}_{gamma}_{seed}".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed))
        # DDPG_model = DDPG.load("models/ddpg_all_arms_action_reward3")
        # ddpg_victim_reward, ddpg_victim_cum_regret, ddpg_attacker_reward = test_attacker(DDPG_model, victim)


def test_attackers(victim, victim_name, gamma):

    for seed in range(0, NUM_SEEDS):
        PPO_model = PPO.load("models/ppo/{victim_name}/{gamma}/ppo_{victim_name}_{gamma}_{seed}".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed), victim)
        victim_rewards, attacker_rewards, victim_regrets, arm_choices = test_attacker(PPO_model, victim, NUM_ITER, NUM_TESTS)
        fname = "raw_results/ppo/{victim_name}/{gamma}/ppo_{victim_name}_{gamma}_{seed}.npz".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        np.savez_compressed(fname, victim_rewards=victim_rewards, attacker_rewards=attacker_rewards, victim_regrets=victim_regrets, arm_choices=arm_choices)

        SAC_model = SAC.load("models/sac/{victim_name}/{gamma}/sac_{victim_name}_{gamma}_{seed}".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed), victim)
        victim_rewards, attacker_rewards, victim_regrets, arm_choices = test_attacker(SAC_model, victim, NUM_ITER, NUM_TESTS)
        "raw_results/sac/{victim_name}/{gamma}/sac_{victim_name}_{gamma}_{seed}.npz".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        np.savez_compressed(fname, victim_rewards=victim_rewards, attacker_rewards=attacker_rewards, victim_regrets=victim_regrets, arm_choices=arm_choices)

        A2C_model = A2C.load("models/a2c/{victim_name}/{gamma}/a2c_{victim_name}_{gamma}_{seed}".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed), victim)
        victim_rewards, attacker_rewards, victim_regrets, arm_choices = test_attacker(A2C_model, victim, NUM_ITER, NUM_TESTS)
        fname = "raw_results/a2c/{victim_name}/{gamma}/a2c_{victim_name}_{gamma}_{seed}.npz".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        np.savez_compressed(fname, victim_rewards=victim_rewards, attacker_rewards=attacker_rewards, victim_regrets=victim_regrets, arm_choices=arm_choices)

        DDPG_model = DDPG.load("models/ddpg/{victim_name}/{gamma}/ddpg_{victim_name}_{gamma}_{seed}".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed), victim)
        victim_rewards, attacker_rewards, victim_regrets, arm_choices = test_attacker(DDPG_model, victim, NUM_ITER, NUM_TESTS)
        fname = "raw_results/ddpg/{victim_name}/{gamma}/ddpg_{victim_name}_{gamma}_{seed}.npz".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        np.savez_compressed(fname, victim_rewards=victim_rewards, attacker_rewards=attacker_rewards, victim_regrets=victim_regrets, arm_choices=arm_choices)


def main():
    environment = env.generate_10_arm_testbed('StandardNormal')
    # bandit = alg.Exp3Algorithm(10, 0.001)
    #bandit = alg.UcbAlgorithm(10)

    victim_exp3 = victims.BanditVictim(alg.Exp3Algorithm(10, 0.001),
                                environment,
                                victims.ObservationModel.ACTION_REWARD,
                                victims.AttackModel.CHOSEN_ARM,
                                np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))

    victim_ucb = victims.BanditVictim(alg.UcbAlgorithm(10),
                                   environment,
                                   victims.ObservationModel.ACTION_REWARD,
                                   victims.AttackModel.CHOSEN_ARM,
                                   np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))


    ## Uncomment to train all attacker models from scratch
    # train_attackers(victim_exp3, "exp3", 0.99)
    # train_attackers(victim_exp3, "exp3", 1)
    # train_attackers(victim_ucb, "ucb", 0.99)
    # train_attackers(victim_ucb, "ucb", 1)

    ## Uncomment run all tests
    # test_attackers(victim_exp3, "exp3", 1)
    # test_attackers(victim_exp3, "exp3", 0.99)
    # test_attackers(victim_ucb, "ucb", 1)
    # test_attackers(victim_ucb, "ucb", 0.99)

    # ## No attack
    # vanilla_model = ZeroAttack(victim)
    # vanilla_victim_reward, vanilla_victim_cum_regret, vanilla_attacker_reward = test_attacker(vanilla_model, victim)
    #
    # ## Uniform random attack
    # random_attack_model = RandomAttack(victim)
    # random_attack_victim_reward, random_attack_victim_cum_regret, random_attack_attacker_reward = test_attacker(random_attack_model, victim)
    #
    #
    npzfile = np.load('raw_results/ppo/exp3/99/ppo_exp3_99_0.npz')
    mean = npzfile['victim_regrets'].mean(axis=0)
    regret = np.cumsum(mean)
    plt.plot(regret)
    plt.show()
    #
    # ## PLOTTING
    # ## Regret plots
    # fig = plt.figure(figsize=(10,6))
    # plt.plot(vanilla_victim_cum_regret, color=palette.DEFAULT_COLORS[0])
    # plt.plot(random_attack_victim_cum_regret, color=palette.DEFAULT_COLORS[1])
    # plt.plot(ddpg_victim_cum_regret, color=palette.DEFAULT_COLORS[2])
    # plt.plot(a2c_victim_cum_regret, color=palette.DEFAULT_COLORS[3])
    # # plt.plot(ppo_victim_cum_regret, color=palette.DEFAULT_COLORS[4])
    # # plt.plot(sac_victim_cum_regret, color=palette.DEFAULT_COLORS[5])
    # plt.xlabel('iterations')
    # plt.ylabel('regret')
    # plt.title('Victim-Level Cumulative Regret')
    # plt.grid()
    # plt.legend(('No attack', 'Uniform random', 'DDPG', 'PPO', 'SAC'))
    # plt.show()



if __name__ == '__main__':
    main()