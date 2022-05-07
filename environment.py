from abc import ABC, abstractmethod
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib.axes as axs
from matplotlib import collections  as mc
from matplotlib import cm
import palette

class Arm(ABC):
    @property
    @abstractmethod
    def support(self):
        pass

    @property
    @abstractmethod
    def mean(self):
        pass

    @property
    def sigma(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def get_pdf(self, start, stop, num_points):
        pass


class GaussianArm(Arm):
    def __init__(self, mean, sigma):
        self._mean = mean
        self._sigma = sigma

    @property
    def support(self):
        return [-np.infty, np.infty]

    @property
    def mean(self):
        return self._mean

    @property
    def sigma(self):
        return self._sigma

    def pull(self):
        return np.random.normal(loc=self._mean, scale=self._sigma)

    def get_pdf(self, start, stop, num_points):
        points = np.linspace(start, stop, num_points)
        return np.exp(- np.square((points - self._mean) / self._sigma) / 2) / np.sqrt(2 * np.pi * self._sigma)


class ExponentialArm(Arm):
    def __init__(self, lamb):
        self._lamb = lamb

    @property
    def support(self):
        return [0, np.infty]

    @property
    def mean(self):
        return 1 / self._lamb

    @property
    def sigma(self):
        return 1 / (self._lamb * self._lamb)

    def pull(self):
        return np.random.exponential(1 / self._lamb)

    def get_pdf(self, start, stop, num_points):
        points = np.linspace(start, stop, num_points)
        return self._lamb * np.exp(- self._lamb * points)


class GammaArm(Arm):
    def __init__(self, shape, rate):
        self._shape = shape  # k
        self._rate = rate    # beta

    @property
    def support(self):
        return [0, np.infty]

    @property
    def mean(self):
        return self._shape / self._rate

    @property
    def sigma(self):
        return self._shape / (self._rate * self._rate)

    def pull(self):
        return np.random.gamma(self._shape, 1 / self._rate)

    def get_pdf(self, start, stop, num_points):
        points = np.linspace(start, stop, num_points)
        return (self._rate ** self._shape) * (points ** (self._shape - 1)) * np.exp(- self._rate * points) / (gamma(self._shape))

class Environment:
    def __init__(self):
        self._arms = []
        self._best_arm = None
        self._last_pulled_arm = None

    @property
    def arms(self):
        return self._arms

    @property
    def best_arm(self):
        return self._best_arm

    @property
    def last_pulled_arm(self):
        return self._last_pulled_arm

    @property
    def arm_means(self):
        return list(map(lambda x: x.mean, self.arms))

    @property
    def num_arms(self):
        return len(self._arms)

    def pull_arm(self, a: int):
        self._last_pulled_arm = self._arms[a]
        reward = self._arms[a].pull()
        return reward, self._last_pulled_arm

    def add_arm(self, arm: Arm):
        self.arms.append(arm)
        if self.best_arm is None:
            self._best_arm = 0
        elif self.arms[self.best_arm].mean < arm.mean:
            self._best_arm = len(self._arms) - 1

    def violin_plot(self, ax: axs.Axes, cmap):
        assert len(cmap) >= len(self._arms), 'The number of colors is less than the number of arms'

        num_samples = 200

        fill_opacity = 0.5
        mean_stem_length = 0.7
        end_stem_length = 0.4
        max_violin_width = 0.8

        interval = 3  # 2 sigma min/max interval
        quantile = 1  # 1 sigma quantile

        axis_stems = []
        mean_stems = []
        end_stems = []
        for i in range(len(self.arms)):
            ## Generating axis lines
            axis_stem = [(i, max(self.arms[i].mean - self.arms[i].sigma * interval, self.arms[i].support[0])),
                         (i, min(self.arms[i].mean + self.arms[i].sigma * interval, self.arms[i].support[1]))]
            mean_stem = [(i - mean_stem_length / 2, self.arms[i].mean),
                         (i + mean_stem_length / 2, self.arms[i].mean)]
            end_stem1 = [(i - end_stem_length / 2, max(self.arms[i].mean - self.arms[i].sigma * interval, self.arms[i].support[0])),
                         (i + end_stem_length / 2, max(self.arms[i].mean - self.arms[i].sigma * interval, self.arms[i].support[0]))]
            end_stem2 = [(i - end_stem_length / 2, min(self.arms[i].mean + self.arms[i].sigma * interval, self.arms[i].support[1])),
                         (i + end_stem_length / 2, min(self.arms[i].mean + self.arms[i].sigma * interval, self.arms[i].support[1]))]
            axis_stems.append(axis_stem)
            mean_stems.append(mean_stem)
            end_stems.append(end_stem1)
            end_stems.append(end_stem2)

            ## Plotting "violin"
            ys = np.linspace(max(self.arms[i].mean - self.arms[i].sigma * interval, self.arms[i].support[0]),
                             min(self.arms[i].mean + self.arms[i].sigma * interval, self.arms[i].support[1]),
                             num_samples)
            xs = self.arms[i].get_pdf(max(self.arms[i].mean - self.arms[i].sigma * interval, self.arms[i].support[0]),
                                      min(self.arms[i].mean + self.arms[i].sigma * interval, self.arms[i].support[1]),
                                      num_samples)

            ax.fill_betweenx(ys, i - xs / 2, i + xs / 2, color=palette.DEFAULT_COLORS[i], alpha=fill_opacity)
            ax.plot(i + xs / 2, ys, color=palette.DEFAULT_COLORS[i], linewidth=1, zorder=2)
            ax.plot(i - xs / 2, ys, color=palette.DEFAULT_COLORS[i], linewidth=1, zorder=2)



        ## Plotting axis lines
        lc = mc.LineCollection(axis_stems, colors=palette.DEFAULT_GRAY, linewidths=1, zorder=3)
        ax.add_collection(lc)
        lc = mc.LineCollection(mean_stems, colors=palette.DEFAULT_GRAY, linewidths=1, zorder=3)
        ax.add_collection(lc)
        lc = mc.LineCollection(end_stems, colors=palette.DEFAULT_GRAY, linewidths=1, zorder=3)
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)


def generate_10_arm_testbed(type = 'StandardNormal'):
    np.random.seed(1)
    env = Environment()
    for i in range(10):
        if type == 'StandardNormal':
            env.add_arm(GaussianArm(np.random.normal(0, 1), 1))
        elif type == 'MixedNormal':
            env.add_arm(GaussianArm(np.random.normal(0, 1), np.random.uniform(0.5, 1)))
        elif type == 'Exponential':
            env.add_arm(ExponentialArm(np.random.uniform(0.5, 2)))
        elif type == 'Gamma':
            env.add_arm(GammaArm(np.random.uniform(1, 4), np.random.uniform(0.5, 2)))
        elif type == 'Mixed':
            r = np.random.rand()
            if r < 0.5:
                env.add_arm(GaussianArm(np.random.uniform(1, 3), np.random.uniform(0.5, 1)))
            else:
                env.add_arm(GammaArm(np.random.uniform(2, 6), 2))

    return env


def test():
    env = generate_10_arm_testbed('StandardNormal')
    print(env.arm_means)
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes((0.1,0.1,0.8,0.8))
    env.violin_plot(ax, palette.DEFAULT_COLORS)
    plt.show()

test()
# test()
