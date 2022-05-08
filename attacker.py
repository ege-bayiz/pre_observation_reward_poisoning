import gym
import bandit_algorithms as alg
import environment as env
import victims
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import palette
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
        np.random.seed(1)
        self.victim = victim

    def victim(self):
        return self.victim

    def predict(self, obs):
        return self.victim.action_space.sample(), 0


def test_attacker(model, victim):
    np.random.seed(1)

    num_iter = 10000
    num_tests = 30

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

    ## Uniform random attack
    random_attack_model = RandomAttack(victim)
    random_attack_victim_reward, random_attack_victim_cum_regret, random_attack_attacker_reward = test_attacker(random_attack_model, victim)

    ## PPO
    PPO_model = PPO('MlpPolicy',
                    victim,
                    verbose=1)
    PPO_model.learn(total_timesteps=1000000)
    PPO_model.save("models/ppo_chosen_arm_action_reward")
    PPO_model = PPO.load("models/ppo_chosen_arm_action_reward", victim)
    ppo_victim_reward, ppo_victim_cum_regret, ppo_attacker_reward = test_attacker(PPO_model, victim)

    ## SAC
    SAC_model = SAC('MlpPolicy',
                    victim,
                    verbose=1,
                    ent_coef='auto_0.1')
    SAC_model.learn(total_timesteps=1000000,
                    log_interval=100)
    SAC_model.save("models/sac_chosen_arm_action_reward")
    SAC_model = SAC.load("models/sac_chosen_arm_action_reward", victim)
    sac_victim_reward, sac_victim_cum_regret, sac_attacker_reward = test_attacker(SAC_model, victim)

    ## A2C
    A2C_model = A2C('MlpPolicy',
                    victim,
                    verbose=1,
                    gamma=0.99)
    A2C_model.learn(total_timesteps=1000000,
                    log_interval=100)
    A2C_model.save("models/a2c_chosen_arm_action_reward")
    A2C_model = A2C.load("models/a2c_chosen_arm_action_reward")
    a2c_victim_reward, a2c_victim_cum_regret, a2c_attacker_reward = test_attacker(A2C_model, victim)

    ## DDPG
    DDPG_model = DDPG('MlpPolicy',
                    victim,
                    verbose=1,
                    gamma=0.99)
    DDPG_model.learn(total_timesteps=1000000,
                    log_interval=100)
    DDPG_model.save("models/ddpg_chosen_arm_action_reward")
    DDPG_model = DDPG.load("models/ddpg_chosen_arm_action_reward")
    ddpg_victim_reward, ddpg_victim_cum_regret, ddpg_attacker_reward = test_attacker(DDPG_model, victim)


    ## PLOTTING
    ## Regret plots
    fig = plt.figure(figsize=(10,6))
    plt.plot(vanilla_victim_cum_regret, color=palette.DEFAULT_COLORS[0])
    plt.plot(random_attack_victim_cum_regret, color=palette.DEFAULT_COLORS[1])
    plt.plot(ddpg_victim_cum_regret, color=palette.DEFAULT_COLORS[2])
    plt.plot(a2c_victim_cum_regret, color=palette.DEFAULT_COLORS[3])
    plt.plot(ppo_victim_cum_regret, color=palette.DEFAULT_COLORS[4])
    plt.plot(sac_victim_cum_regret, color=palette.DEFAULT_COLORS[5])
    plt.xlabel('iterations')
    plt.ylabel('regret')
    plt.title('Victim-Level Cumulative Regret')
    plt.grid()
    plt.legend(('No attack', 'Uniform random', 'DDPG', 'A2C', 'PPO', 'SAC'))
    plt.show()


if __name__ == '__main__':
    main()