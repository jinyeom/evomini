import numpy as np

class ES:
  def __init__(self, num_params,
               sigma=0.1,
               stepsize=0.01,
               beta1=0.99,
               beta2=0.999):
    self.num_params = num_params
    self.mu = np.zeros(num_params)
    self.sigma = sigma
    self.stepsize = stepsize
    self.beta1 = beta1
    self.beta2 = beta2
    self.t = 0
    self.m = np.zeros_like(self.mu)
    self.v = np.zeros_like(self.mu)
    self.epsilon = None

  def sample(self, popsize):
    # antithetic/symmetric sampling
    assert popsize % 2 == 0
    eps_split = np.random.randn(popsize // 2, self.num_params)
    self.epsilon = np.concatenate([eps_split, -eps_split], axis=0)
    env_seeds = np.random.randint(2 ** 31 - 1, size=popsize, dtype=int)
    solutions = self.mu + self.sigma * self.epsilon
    return env_seeds, solutions

  def step(self, fitness):
    assert self.epsilon is not None
    assert len(fitness.shape) == 1
    popsize = fitness.shape[0]
    # shape fitness values as normalized ranks (higher the better)
    rank = np.empty_like(fitness, dtype=np.long)
    rank[np.argsort(fitness)] = np.arange(popsize)
    rank = rank.astype(np.float) / (popsize - 1) - 0.5
    rank = (rank - np.mean(rank)) / np.std(rank)
    grad = 1 / (popsize * self.sigma) * (self.epsilon.T @ rank)
    # gradient ascent with Adam
    self.t += 1
    self.m = self.beta1 * self.m + (1 - self.beta1) * grad
    self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
    a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
    step = a * self.m / (np.sqrt(self.v) + 1e-8)
    ratio = np.linalg.norm(step) / (np.linalg.norm(self.mu) + 1e-8)
    self.mu += step
    return ratio
