import numpy as np

class GA:
  def __init__(self, num_params, sigma=0.1, num_topk=10):
    self.num_params = num_params
    self.sigma = sigma
    self.num_topk = num_topk
    self.population = None
    self.topk = None
    self.elite = None

  def sample(self, popsize):
    env_seeds = np.random.randint(2 ** 31 - 1, size=popsize, dtype=int)
    self.population = self.sigma * np.random.randn(popsize, self.num_params)
    return env_seeds, self.population

  def get_topk(self, fitness):
    assert self.population is not None
    assert len(fitness) == self.population.shape[0]
    population = self.population[np.argsort(fitness)]
    self.topk = population[::-1][:self.num_topk]
    return self.topk

  def get_elite(self, fitness):
    assert self.topk is not None
    assert len(fitness) == self.num_topk
    topk = self.topk[np.argsort(fitness)]
    self.elite = topk[::-1][0]
    return self.elite

  def reproduce(self):
    raise NotImplementedError

if __name__ == "__main__":
  ga = GA(10)
  seeds, solutions = ga.sample(20)
  ga.step(np.random.rand(20))
