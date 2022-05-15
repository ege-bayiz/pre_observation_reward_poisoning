import gym
import bandit_algorithms as alg
import environment as env
import victims
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import numpy as np
import os
import palette
import pandas as pd
import re
from abc import ABC, abstractmethod
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import A2C
from stable_baselines3 import DDPG

NUM_SEEDS = 10
NUM_ITER = 10000
NUM_TESTS = 30
VICTIM_MODELS = {'ucb': alg.UcbAlgorithm,
                 'exp3': alg.Exp3Algorithm}
ATTACKER_MODELS = ["a2c", "ddpg", "sac"]
GAMMA_VALUES = [0.99, 1]
TARGET_ARM = 5

class AttackModel(ABC):
    """Custom attacker model for testing hand designed attack policies"""
    @property
    @abstractmethod
    def victim(self):
        pass

    @abstractmethod
    def predict(self, obs):
        pass

    @abstractmethod
    def set_random_seed(self, seed):
        pass

class ZeroAttack(AttackModel):
    def __init__(self, victim: victims.BanditVictim):
        self.victim = victim

    def victim(self):
        return self.victim

    def predict(self, obs):
        return np.zeros(self.victim.action_space.shape), 0

    def set_random_seed(self, seed):
        pass


class RandomAttack(AttackModel):
    def __init__(self, victim: victims.BanditVictim):
        np.random.seed(1)
        self.victim = victim

    def victim(self):
        return self.victim

    def predict(self, obs):
        return self.victim.action_space.sample(), 0

    def set_random_seed(self, seed):
        np.random.seed(seed)  # the action space sampling in gym is based on numpy rng

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


def train_attackers(victim, victim_name, gamma):
    for seed in range(0, NUM_SEEDS):
        # PPO
        # PPO_model = PPO('MlpPolicy',
        #                 victim,
        #                 verbose=1,
        #                 gamma=gamma,
        #                 seed=seed)
        # PPO_model.learn(total_timesteps=100000)
        # PPO_model.save("models/ppo/{victim_name}/{gamma}/ppo_{victim_name}_{gamma}_{seed}".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed))

        # SAC
        SAC_model = SAC('MlpPolicy',
                        victim,
                        verbose=1,
                        gamma=gamma,
                        seed=seed)
        SAC_model.learn(total_timesteps=100000)
        SAC_model.save("models/sac/{victim_name}/{gamma}/sac_{victim_name}_{gamma}_{seed}".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed))

        # A2C
        A2C_model = A2C('MlpPolicy',
                        victim,
                        verbose=1,
                        gamma=gamma,
                        seed=seed)
        A2C_model.learn(total_timesteps=100000)
        A2C_model.save("models/a2c/{victim_name}/{gamma}/a2c_{victim_name}_{gamma}_{seed}".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed))

        # DDPG
        DDPG_model = DDPG('MlpPolicy',
                          victim,
                          verbose=1,
                          gamma=gamma,
                          seed=seed)
        DDPG_model.learn(total_timesteps=100000, log_interval=1)
        DDPG_model.save("models/ddpg/{victim_name}/{gamma}/ddpg_{victim_name}_{gamma}_{seed}".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed))


def test_attackers(victim, victim_name, gamma):
    # Generates raw results for all attackers
    for seed in range(0, NUM_SEEDS):
        vanilla_model = ZeroAttack(victim)
        victim_rewards, attacker_rewards, victim_regrets, arm_choices = test_attacker(vanilla_model, victim, NUM_ITER, NUM_TESTS)
        fname = "raw_results/vanilla/{victim_name}/{gamma}/vanilla_{victim_name}_{gamma}_{seed}.npz".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        np.savez_compressed(fname, victim_rewards=victim_rewards, attacker_rewards=attacker_rewards, victim_regrets=victim_regrets, arm_choices=arm_choices)

        random_attack_model = RandomAttack(victim)
        victim_rewards, attacker_rewards, victim_regrets, arm_choices = test_attacker(random_attack_model, victim, NUM_ITER, NUM_TESTS)
        fname = "raw_results/random/{victim_name}/{gamma}/random_{victim_name}_{gamma}_{seed}.npz".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        np.savez_compressed(fname, victim_rewards=victim_rewards, attacker_rewards=attacker_rewards, victim_regrets=victim_regrets, arm_choices=arm_choices)

        # PPO_model = PPO.load("models/ppo/{victim_name}/{gamma}/ppo_{victim_name}_{gamma}_{seed}".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed), victim)
        # victim_rewards, attacker_rewards, victim_regrets, arm_choices = test_attacker(PPO_model, victim, NUM_ITER, NUM_TESTS)
        # fname = "raw_results/ppo/{victim_name}/{gamma}/ppo_{victim_name}_{gamma}_{seed}.npz".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed)
        # os.makedirs(os.path.dirname(fname), exist_ok=True)
        # np.savez_compressed(fname, victim_rewards=victim_rewards, attacker_rewards=attacker_rewards, victim_regrets=victim_regrets, arm_choices=arm_choices)

        SAC_model = SAC.load("models/sac/{victim_name}/{gamma}/sac_{victim_name}_{gamma}_{seed}".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed), victim)
        victim_rewards, attacker_rewards, victim_regrets, arm_choices = test_attacker(SAC_model, victim, NUM_ITER, NUM_TESTS)
        fname = "raw_results/sac/{victim_name}/{gamma}/sac_{victim_name}_{gamma}_{seed}.npz".format(victim_name=victim_name, gamma=int(gamma * 100), seed=seed)
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


def generate_result_table():
    # Generates a result table that summarized the performance of attackers
    columns = ["attacker_name",
               "victim_name",
               "gamma",
               "min_victim_regret",
               "mean_victim_regret",
               "max_victim_regret",
               "std_victim_regret",
               "min_target_arm_picks",
               "mean_target_arm_picks",
               "max_target_arm_picks",
               "min_target_arm_picks_(last_1000_steps)",
               "mean_target_arm_picks_(last_1000_steps)",
               "max_target_arm_picks_(last_1000_steps)",
               "best_seed",
               "worst_seed"]
    df = pd.DataFrame(columns=columns)

    # First generate results for random attack and no attack as baselines
    for attacker_name in tqdm(["vanilla", "random"], position=1, colour='red'):
        for victim_name in tqdm(VICTIM_MODELS.keys(), position=2, colour='blue', leave=False):
            gamma = GAMMA_VALUES[0]  # Discount factor does not matter in these tests
            input_directory = 'raw_results/{attacker_name}/{victim_name}/{gamma}/'.format(
                attacker_name= attacker_name, victim_name=victim_name, gamma=int(gamma * 100))
            gamma = "-"
            cum_regrets = []
            target_arm_picks = []
            target_arm_picks_last = []
            seeds = []
            for filename in tqdm(os.listdir(input_directory), position=3, colour='green', leave=False):
                seed = re.split('_|\.', filename)[3]
                seeds.append(seed)
                f = os.path.join(input_directory, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    npzfile = np.load(f)
                    correct = np.count_nonzero(npzfile['arm_choices'][:,-1000:] == TARGET_ARM, axis=1)
                    cum_regrets.append(np.sum(npzfile['victim_regrets'], axis=1).mean())
                    target_arm_picks.append(np.count_nonzero(npzfile['arm_choices'] == TARGET_ARM, axis=1).mean())
                    target_arm_picks_last.append(np.count_nonzero(npzfile['arm_choices'][:,-1000:] == TARGET_ARM, axis=1).mean())


            cum_regrets = np.asarray(cum_regrets)
            target_arm_picks = np.asarray(target_arm_picks)
            target_arm_picks_last = np.asarray(target_arm_picks_last)
            best_seed = seeds[target_arm_picks_last.argmax()]
            worst_seed = seeds[target_arm_picks_last.argmin()]

            results = {"attacker_name": attacker_name,
                       "victim_name": victim_name,
                       "gamma": gamma,
                       "min_victim_regret": cum_regrets.min(),
                       "mean_victim_regret": cum_regrets.mean(),
                       "max_victim_regret": cum_regrets.max(),
                       "std_victim_regret": cum_regrets.std(),
                       "min_target_arm_picks": target_arm_picks.min(),
                       "mean_target_arm_picks": target_arm_picks.mean(),
                       "max_target_arm_picks": target_arm_picks.max(),
                       "min_target_arm_picks_(last_1000_steps)": target_arm_picks_last.min(),
                       "mean_target_arm_picks_(last_1000_steps)": target_arm_picks_last.mean(),
                       "max_target_arm_picks_(last_1000_steps)": target_arm_picks_last.max(),
                       "best_seed": best_seed,
                       "worst_seed": worst_seed}

            df = df.append(results, ignore_index=True)

    # Results for the remaining attack models
    for attacker_name in tqdm(ATTACKER_MODELS, position=1, colour='red'):
        for victim_name in tqdm(VICTIM_MODELS.keys(), position=2, colour='blue', leave=False):
            for gamma in GAMMA_VALUES:
                input_directory = 'raw_results/{attacker_name}/{victim_name}/{gamma}/'.format(
                    attacker_name= attacker_name, victim_name=victim_name, gamma=int(gamma * 100))
                cum_regrets = []
                target_arm_picks = []
                target_arm_picks_last = []
                seeds = []
                for filename in tqdm(os.listdir(input_directory), position=3, colour='green', leave=False):
                    seed = re.split('_|\.', filename)[3]
                    seeds.append(seed)
                    f = os.path.join(input_directory, filename)
                    if os.path.isfile(f):
                        npzfile = np.load(f)
                        cum_regrets.append(np.sum(npzfile['victim_regrets'], axis=1).mean())
                        target_arm_picks.append(np.count_nonzero(npzfile['arm_choices'] == TARGET_ARM, axis=1).mean())
                        target_arm_picks_last.append(np.count_nonzero(npzfile['arm_choices'][:,-1000:] == TARGET_ARM, axis=1).mean())

                cum_regrets = np.asarray(cum_regrets)
                target_arm_picks = np.asarray(target_arm_picks)
                target_arm_picks_last = np.asarray(target_arm_picks_last)
                best_seed = seeds[cum_regrets.argmax()]
                worst_seed = seeds[cum_regrets.argmin()]

                results = {"attacker_name": attacker_name,
                           "victim_name": victim_name,
                           "gamma": gamma,
                           "min_victim_regret": cum_regrets.min(),
                           "mean_victim_regret": cum_regrets.mean(),
                           "max_victim_regret": cum_regrets.max(),
                           "std_victim_regret": cum_regrets.std(),
                           "min_target_arm_picks": target_arm_picks.min(),
                           "mean_target_arm_picks": target_arm_picks.mean(),
                           "max_target_arm_picks": target_arm_picks.max(),
                           "min_target_arm_picks_(last_1000_steps)": target_arm_picks_last.min(),
                           "mean_target_arm_picks_(last_1000_steps)": target_arm_picks_last.mean(),
                           "max_target_arm_picks_(last_1000_steps)": target_arm_picks_last.max(),
                           "best_seed": best_seed,
                           "worst_seed": worst_seed}

                df = df.append(results, ignore_index=True)

    df.to_csv('refined_results/results_summary.csv', index=False)


def plot_results():
    ## Font size settings
    font = {'family': 'sans',
            'weight': 'normal',
            'size': 18}
    plt.rcParams["font.family"] = "Times New Roman"
    matplotlib.rc('font', **font)

    # EXP3
    # Selected Attackers/ Best attackers in each class, selected manually based on victim regret and correct arm picks.
    no_attack_npz = np.load('raw_results/vanilla/exp3/99/vanilla_exp3_99_0.npz')
    random_npz = np.load('raw_results/random/exp3/99/random_exp3_99_0.npz')
    a2c_npz = np.load('raw_results/a2c/exp3/99/a2c_exp3_99_1.npz')
    ddpg_npz = np.load('raw_results/ddpg/exp3/100/ddpg_exp3_100_7.npz')
    #ppo_npz = np.load('raw_results/ppo/exp3/99/ppo_exp3_99_6.npz')
    sac_npz = np.load('raw_results/sac/exp3/99/sac_exp3_99_6.npz')

    ## Regret plots (exp3)
    iterations = range(1, no_attack_npz['victim_regrets'].shape[1] + 1)
    sigma = 0.1
    fig = plt.figure(figsize=(10, 6))
    mean = np.cumsum(no_attack_npz['victim_regrets'].mean(axis=0))
    std = np.cumsum(no_attack_npz['victim_regrets'], axis=1).std(axis=0)
    plt.plot(mean, color=palette.DEFAULT_COLORS[0], label="no attack")
    plt.fill_between(iterations, mean - sigma*std, mean + sigma*std, color=palette.DEFAULT_COLORS[0], alpha=0.2)

    mean = np.cumsum(random_npz['victim_regrets'].mean(axis=0))
    std = np.cumsum(random_npz['victim_regrets'], axis=1).std(axis=0)
    plt.plot(mean, color=palette.DEFAULT_COLORS[1], label="random")
    plt.fill_between(iterations, mean - sigma*std, mean + sigma*std, color=palette.DEFAULT_COLORS[1], alpha=0.2)

    mean = np.cumsum(a2c_npz['victim_regrets'].mean(axis=0))
    std = np.cumsum(a2c_npz['victim_regrets'], axis=1).std(axis=0)
    plt.plot(mean, color=palette.DEFAULT_COLORS[2], label="A2C")
    plt.fill_between(iterations, mean - sigma*std, mean + sigma*std, color=palette.DEFAULT_COLORS[2], alpha=0.2)

    mean = np.cumsum(ddpg_npz['victim_regrets'].mean(axis=0))
    std = np.cumsum(ddpg_npz['victim_regrets'], axis=1).std(axis=0)
    plt.plot(mean, color=palette.DEFAULT_COLORS[3], label="DDPG")
    plt.fill_between(iterations, mean - sigma*std, mean + sigma*std, color=palette.DEFAULT_COLORS[3], alpha=0.2)

    # mean = np.cumsum(ppo_npz['victim_regrets'].mean(axis=0))
    # std = np.cumsum(ppo_npz['victim_regrets'], axis=1).std(axis=0)
    # plt.plot(mean, color=palette.DEFAULT_COLORS[4], label="PPO")
    # plt.fill_between(iterations, mean - sigma*std, mean + sigma*std, color=palette.DEFAULT_COLORS[4], alpha=0.2)

    mean = np.cumsum(sac_npz['victim_regrets'].mean(axis=0))
    std = np.cumsum(sac_npz['victim_regrets'], axis=1).std(axis=0)
    plt.plot(mean, color=palette.DEFAULT_COLORS[5], label="SAC")
    plt.fill_between(iterations, mean - sigma*std, mean + sigma*std, color=palette.DEFAULT_COLORS[5], alpha=0.2)

    plt.xlabel('iterations')
    plt.ylabel('regret')
    plt.title('Victim-Level Cumulative Regret')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.margins(0)
    plt.savefig('refined_results/plots/exp3_victim_regrets.pdf')

    ## Attacker Reward Plots
    fig = plt.figure(figsize=(10, 6))
    plt.plot(random_npz['attacker_rewards'].mean(axis=0), color=palette.DEFAULT_COLORS[1], label="random")
    plt.plot(a2c_npz['attacker_rewards'].mean(axis=0), color=palette.DEFAULT_COLORS[2], label="A2C")
    plt.plot(ddpg_npz['attacker_rewards'].mean(axis=0), color=palette.DEFAULT_COLORS[3], label="DDPG")
    # plt.plot(ppo_npz['attacker_rewards'].mean(axis=0), color=palette.DEFAULT_COLORS[4], label="PPO")
    plt.plot(sac_npz['attacker_rewards'].mean(axis=0), color=palette.DEFAULT_COLORS[5], label="SAC")
    plt.xlabel('iterations')
    plt.ylabel('reward')
    plt.title('Attacker-Level Reward')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.margins(0)
    plt.savefig('refined_results/plots/exp3_attacker_rewards.pdf')

    # UCB -------------------------------------------------------------------------------------------------------------
    # Selected Attackers/ Best attackers in each class, selected manually based on victim regret and correct arm picks.
    no_attack_npz = np.load('raw_results/vanilla/ucb/99/vanilla_ucb_99_0.npz')
    random_npz = np.load('raw_results/random/ucb/99/random_ucb_99_5.npz')
    a2c_npz = np.load('raw_results/a2c/ucb/100/a2c_ucb_100_3.npz')
    ddpg_npz = np.load('raw_results/ddpg/ucb/100/ddpg_ucb_100_6.npz')
    sac_npz = np.load('raw_results/sac/ucb/100/sac_ucb_100_5.npz')
    #ppo_npz = np.load('raw_results/ppo/ucb/100/ppo_ucb_100_5.npz')

    ## Regret plots (ucb)
    iterations = range(1, no_attack_npz['victim_regrets'].shape[1] + 1)
    sigma = 0.1
    fig = plt.figure(figsize=(10, 6))
    mean = np.cumsum(no_attack_npz['victim_regrets'].mean(axis=0))
    std = np.cumsum(no_attack_npz['victim_regrets'], axis=1).std(axis=0)
    plt.plot(mean, color=palette.DEFAULT_COLORS[0], label="no attack")
    plt.fill_between(iterations, mean - sigma * std, mean + sigma * std, color=palette.DEFAULT_COLORS[0], alpha=0.2)

    mean = np.cumsum(random_npz['victim_regrets'].mean(axis=0))
    std = np.cumsum(random_npz['victim_regrets'], axis=1).std(axis=0)
    plt.plot(mean, color=palette.DEFAULT_COLORS[1], label="random")
    plt.fill_between(iterations, mean - sigma * std, mean + sigma * std, color=palette.DEFAULT_COLORS[1], alpha=0.2)

    mean = np.cumsum(a2c_npz['victim_regrets'].mean(axis=0))
    std = np.cumsum(a2c_npz['victim_regrets'], axis=1).std(axis=0)
    plt.plot(mean, color=palette.DEFAULT_COLORS[2], label="A2C")
    plt.fill_between(iterations, mean - sigma * std, mean + sigma * std, color=palette.DEFAULT_COLORS[2], alpha=0.2)

    mean = np.cumsum(ddpg_npz['victim_regrets'].mean(axis=0))
    std = np.cumsum(ddpg_npz['victim_regrets'], axis=1).std(axis=0)
    plt.plot(mean, color=palette.DEFAULT_COLORS[3], label="DDPG")
    plt.fill_between(iterations, mean - sigma * std, mean + sigma * std, color=palette.DEFAULT_COLORS[3], alpha=0.2)

    # mean = np.cumsum(ppo_npz['victim_regrets'].mean(axis=0))
    # std = np.cumsum(ppo_npz['victim_regrets'], axis=1).std(axis=0)
    # plt.plot(mean, color=palette.DEFAULT_COLORS[4],  label="PPO")
    # plt.fill_between(iterations, mean - sigma*std, mean + sigma*std, color=palette.DEFAULT_COLORS[4], alpha=0.2)

    mean = np.cumsum(sac_npz['victim_regrets'].mean(axis=0))
    std = np.cumsum(sac_npz['victim_regrets'], axis=1).std(axis=0)
    plt.plot(mean, color=palette.DEFAULT_COLORS[5], label="SAC")
    plt.fill_between(iterations, mean - sigma * std, mean + sigma * std, color=palette.DEFAULT_COLORS[5], alpha=0.2)

    plt.xlabel('iterations')
    plt.ylabel('regret')
    plt.title('Victim-Level Cumulative Regret')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.margins(0)
    plt.savefig('refined_results/plots/ucb_victim_regrets.pdf')

    ## Attacker Reward Plots
    fig = plt.figure(figsize=(10, 6))
    plt.plot(random_npz['attacker_rewards'].mean(axis=0), color=palette.DEFAULT_COLORS[1], label="random")
    plt.plot(a2c_npz['attacker_rewards'].mean(axis=0), color=palette.DEFAULT_COLORS[2], label="A2C")
    plt.plot(ddpg_npz['attacker_rewards'].mean(axis=0), color=palette.DEFAULT_COLORS[3], label="DDPG")
    # plt.plot(ppo_npz['attacker_rewards'].mean(axis=0), color=palette.DEFAULT_COLORS[4], label="PPO")
    plt.plot(sac_npz['attacker_rewards'].mean(axis=0), color=palette.DEFAULT_COLORS[5], label="SAC")
    plt.xlabel('iterations')
    plt.ylabel('reward')
    plt.title('Attacker-Level Reward')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.margins(0)
    plt.savefig('refined_results/plots/ucb_attacker_rewards.pdf')


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

    # generate_result_table()
    plot_results()

if __name__ == '__main__':
    main()