from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import palette
from scipy.special import softmax
import environment
from environment import Environment
from tqdm import tqdm

class BanditAlgorithm(ABC):
    def __init__(self, num_arms):
        self._num_arms = num_arms

    @property
    def num_arms(self):
        return self._num_arms

    @abstractmethod
    def choose_arm(self) -> int:
        pass

    @abstractmethod
    def update(self, arm, reward):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_arm_probs(self):
        pass


class Exp3Algorithm(BanditAlgorithm):
    """
    Exp3 Algorithm
    Algorithm 9 in Tor Lattimore, Csaba Szepesvari "Bandit Algorithms" online edition.
    """
    def __init__(self, num_arms, learning_rate):
        super().__init__(num_arms)
        self._learning_rate = learning_rate
        self._P = np.zeros(num_arms)
        self._S = np.zeros(num_arms)
        self.reset()

    def choose_arm(self):
        return int(np.random.choice(self.num_arms, p=self._P))

    def update(self, arm, reward):
        indicator = np.zeros(self.num_arms)
        indicator[arm] = 1

        S_update = ((indicator * (1 - reward)) / self._P)
        if np.isnan(S_update).any():
            S_update[np.isnan(S_update)] = 0
        self._S = self._S - S_update
        self._S -= np.max(self._S)  # for numerical stability
        self._P = softmax(self._learning_rate * self._S)

    def reset(self):
        self._S = np.zeros(self._num_arms)
        exp_sum = np.sum(np.exp(self._learning_rate * self._S))
        self._P = np.exp(self._learning_rate * self._S) / exp_sum

    def get_arm_probs(self):
        return self._P


class UcbAlgorithm(BanditAlgorithm):
    """
    Exp3 Algorithm
    Algorithm 6 in Tor Lattimore, Csaba Szepesvari "Bandit Algorithms" online edition.
    """
    def __init__(self, num_arms):
        super().__init__(num_arms)
        self._time = 0.0
        self._emp_means = np.zeros(self.num_arms)
        self._num_pulls = np.zeros(self.num_arms)
        self.reset()

    def choose_arm(self):
        if self._time < self.num_arms:
            return int(self._time)
        else:
            return self._get_best_arm()


    def _get_best_arm(self):
        return np.argmax(self._emp_means + self._compute_ucb_bonus())

    def _compute_ucb_bonus(self):
        f = 1 + self._time * (np.log(self._time) ** 2)
        return np.sqrt(2 * np.log(f) / self._num_pulls)

    def update(self, arm, reward):
        self._emp_means[arm] = (self._emp_means[arm] * self._num_pulls[arm] + reward) \
                              / (self._num_pulls[arm] + 1)  # update empirical means
        self._num_pulls[arm] += 1
        self._time += 1

    def reset(self):
        self._time = 0.0
        self._emp_means = np.zeros(self.num_arms)
        self._num_pulls = np.zeros(self.num_arms)

    def get_arm_probs(self):
        p = np.zeros(self.num_arms)
        p[self.choose_arm()] = 1
        return p


def test_bandit_algorithm(alg: BanditAlgorithm, env: Environment, num_iter: int, num_tests: int):
    avg_reward = np.zeros(num_iter)
    avg_cum_regret = np.zeros(num_iter)
    for k in tqdm(range(num_tests), position=1, colour='red'):
        cum_regret = 0
        alg.reset()
        for t in tqdm(range(num_iter), position=2, colour='blue', leave=False):
            arm = alg.choose_arm()
            reward, _ = env.pull_arm(arm)
            alg.update(arm=arm, reward=reward)

            avg_reward[t] += 1/(k+1) * (reward - avg_reward[t])

            best_reward, _ = env.pull_arm(env.best_arm)

            cum_regret += best_reward - reward
            avg_cum_regret[t] += 1/(k+1) * (cum_regret - avg_cum_regret[t])


    fig = plt.figure(figsize=(10, 6))
    plt.plot(avg_cum_regret, color=palette.DEFAULT_COLORS[0])

    plt.xlabel('iterations')
    plt.ylabel('regret')
    plt.title('Cumulative Regret')


# env = environment.generate_10_arm_testbed('StandardNormal')
# bandit = Exp3Algorithm(10, 0.001)
#
# env = environment.generate_10_arm_testbed('StandardNormal')
# bandit = UcbAlgorithm(10)

# test_bandit_algorithm(bandit, env, 10000, 100)