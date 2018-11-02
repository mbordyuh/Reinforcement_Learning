'''Solving exploit - explore delema for 3 bandits using
epsilon greedy approach'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, m):
        self.m = m  # True Bandit mean
        self.mean_ = 0  # Estimate of Bandits mean
        self.N = 0

    def pull(self):
        # Sample from Gaussian with unit variance around bandits true mean
        return np.random.rand() + self.m

    def update(self, x):
        # Update estimated mean value of a bandit after a pull
        self.N += 1
        self.mean_ = (1 - 1/self.N)*self.mean_ + 1/self.N*x


def run_experiment(m1, m2, m3, eps, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    data = np.empty(N)

    for i in range(N):
        p = np.random.rand()
        if p < eps:
            j = np.random.choice(3)
            x = bandits[j].pull()
            bandits[j].update(x)
        else:
            j = np.argmax([bandit.mean_ for bandit in bandits])
            x = bandits[j].pull()
            bandits[j].update(x)

        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    # Plot cumulative average
    plt.figure()
    plt.plot(cumulative_average)
    plt.ylabel('Cumulative average')
    plt.xlabel('N')
    plt.xscale('log')
    plt.legend([f'epsilon = {eps}'])
    plt.show()


if __name__ == 'main':
    run_experiment(1, 2, 3, 0.01, 100000)
