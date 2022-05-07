from collections import deque
import numpy as np

INIT_TRUST = 1

class Sanitizer:
    def __init__(self, num_arms = 10, length = 10, delta = 1, total_energy = 1):
        self.max_len = length
        self.total_energy = total_energy
        self.arm_estimates = np.zeros(num_arms)
        self.num_arms = num_arms
        self.delta = delta
        self.data_buffer = deque()
        self.data_out = None
        self.trust_buffer = deque()
        self.trust_out = None

    def push(self, x):
        self.data_buffer.append(x)
        self.trust_buffer.append(INIT_TRUST)
        if len(self.data_buffer) > self.max_len:
            self.data_out = self.data_buffer.popleft()
            self.trust_out = self.trust_buffer.popleft()
        self._sanitize()

    def set_arm_estimates(self, mu):
        self.arm_estimates = mu

    def out(self):
        return self.data_out, self.trust_out


    def _sanitize(self):

        pass

def get_reward(avg, var):
    rew = np.zeros(avg.size)
    cov = np.identity(avg.size) * var
    rew = np.random.multivariate_normal(avg, cov)

    return rew


def add_adversarial_noise():
    pass


class EXP3(object):
    def __init__(self, avg, eta):  ## Initialization

        self.means = avg
        self.num_arms = avg.size
        self.eta = eta
        self.best_arm = np.argmax(self.means)
        self.restart()

        return None

        ## variable names (most are self explanatory)
        ## self.num_arms is the number of arms (k)
        ## self.means[arm] is the vector of true means of the arms
        ## self.time is the current time index
        ## self.best_arm is the best arm given the true mean rewards
        ## self.cum_reg is the cumulative regret
        ## self.num_plays[arm] is the vector of number of times that arm k has been pulled
        ## self.eta is the learning rate
        ## self.probs_arr is the sampling distribution vector P_t
        ## self.S is the vector of estimated reward by the end of time t

    def restart(self):  ## Restart the algorithm: Reset self.time to zero (done)
        self.time = 0.0
        self.S = np.zeros(self.num_arms)
        self.emp_means = np.zeros(self.num_arms)
        self.num_plays = np.zeros(self.num_arms)
        self.cum_reg = [0]
        self.probs_arr = np.ones(self.num_arms) / self.num_arms
        return None

    def get_best_arm(self):  ## For each time index, find the best arm according to EXP3
        return int(np.random.choice(self.num_arms, p=self.probs_arr))

    def update_exp3(self, arm, rew_vec):  ## Compute probs_arr and update the total estimated reward for each arm
        indicator = np.zeros(self.num_arms)
        indicator[arm] = 1

        self.S = self.S + 1 - ((indicator * (1 - rew_vec)) / self.probs_arr)
        self.probs_arr = np.exp(self.eta * self.S) / np.sum(np.exp(self.eta * self.S))
        return None

    def update_reg(self, arm, rew_vec):  ## Update the cumulative regret vector

        self.cum_reg = np.append(self.cum_reg, self.cum_reg[-1] + (rew_vec[self.best_arm] - rew_vec[arm]))
        return None

    def iterate(self, rew_vec):  ## Iterate the algorithm
        self.time += 1.0
        arm = self.get_best_arm()
        self.update_exp3(arm, rew_vec)
        self.update_reg(arm, rew_vec)

        ## Your Code here

        return None


def run_algo(avg, eta, num_iter, num_inst, var):
    reg = np.zeros((num_inst, num_iter))

    algo = EXP3(avg, eta)

    for k in range(num_inst):
        algo.restart()

        if (k + 1) % 10 == 0:
            print('Instance number = ', k + 1)

        for t in range(num_iter - 1):
            rew_vec = get_reward(avg, var)
            if 500 < t < 1000:
                e = np.zeros(len(avg))
                e[4] = 0.3
                rew_vec += e

            algo.iterate(rew_vec)

        reg[k, :] = np.asarray(algo.cum_reg)

    return reg

def main():
    avg = np.asarray([0.8, 0.7, 0.5, 0.65, 0.15, 0.75, 0.7, 0.2, 0.5, 0.55])
    num_iter, num_inst = int(1e4), 20
    eta = np.sqrt(np.log(avg.size) / (num_iter * avg.size))
    var = 0.01

    reg = run_algo(avg, eta, num_iter, num_inst, var)

    ## Your Code here
    import matplotlib.pyplot as plt
    plt.plot(range(num_iter), np.mean(reg, axis=0))
    plt.legend(('Cumulative Regret', 'x = len(avg)*m'))
    plt.xlabel('time')
    plt.ylabel('$Cumulative Regret$')
    plt.title('Cumulative Regret with EXP$3$')
    plt.grid()

    ax2 = plt.figure()
    plt.semilogx(range(num_iter), np.mean(reg, axis=0))
    plt.title('Cumulative Regret with EXP3 (Logarithmic time axis)')
    plt.xlabel('t')
    plt.ylabel('$R_t$')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
