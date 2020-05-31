from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Individual:
  genome: np.ndarray
  fitness: Optional[float] = None
  is_elite: bool = False

  def clone(self, mut_sigma):
    # return a mutated child
    child = Individual(np.array(self.genome))
    child.genome += mut_sigma * np.random.randn(len(child.genome))
    return child

class SimpleGA:
  def __init__(self, num_params,
               sigma=0.1,
               topk=10,
               trunc_thresh=100,
               sample_size=2,
               mut_sigma=0.01):
    self.num_params = num_params
    self.sigma = sigma
    self.topk = topk
    self.trunc_thresh = trunc_thresh
    self.sample_size = sample_size
    self.mut_sigma = mut_sigma
    self.population = None
    self.candidates = None
    self.elite = None

  def sample(self, popsize):
    env_seeds = np.random.randint(2 ** 31 - 1, size=popsize, dtype=int)
    solutions = self.sigma * np.random.randn(popsize, self.num_params)
    self.population = [Individual(genome) for genome in solutions]
    return env_seeds, solutions

  def set_elite_candidates(self, fitness):
    assert self.population is not None
    for ind, fit in zip(self.population, fitness):
      ind.fitness = fit
    self.population = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
    self.candidates = self.population[:self.topk]
    # return top K solutions and environment seeds for the second evaluation
    env_seeds = np.random.randint(2 ** 31 - 1, size=self.topk, dtype=int)
    solutions = np.array([ind.genome for ind in self.candidates])
    return env_seeds, solutions

  def set_elite(self, fitness):
    assert self.candidates is not None
    # sort the elite candidates by their second fitness values
    self.candidates = [ind for ind, _ in sorted(zip(self.candidates, fitness),
                      key=lambda ind_fit_tuple: ind_fit_tuple[1], reverse=True)]
    self.elite = self.candidates[0]
    self.elite.is_elite = True
    return np.array(self.elite.genome)

  def reproduce(self):
    assert self.elite is not None
    if len(self.population) > self.trunc_thresh - 1:
      del self.population[self.trunc_thresh - 1:]
    self.population.append(self.elite)
    offsprings = []
    for i in range(self.trunc_thresh):
      samples = np.random.choice(self.population, size=self.sample_size)
      samples = sorted(samples, key=lambda ind: ind.fitness, reverse=True)
      offspring = samples[0].clone(self.mut_sigma)
      offsprings.append(offspring)
    self.population.extend(offsprings)
    # return the next generation environment seeds and solutions
    env_seeds = np.random.randint(2 ** 31 - 1, size=len(self.population), dtype=int)
    solutions = np.array([ind.genome for ind in self.population])
    return env_seeds, solutions
