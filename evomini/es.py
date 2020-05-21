import math
import multiprocessing as mp
import numpy as np

class Evaluator:
  def __init__(self, num_workers,
               models_per_worker,
               num_evals,
               precision):
    self.num_workers = num_workers
    self.models_per_worker = models_per_worker
    self.num_evals = num_evals
    self.precision = precision

    self.pipes = []
    self.procs = []
    for rank in range(self.num_workers):
      parent_pipe, child_pipe = mp.Pipe()
      proc = mp.Process(target=self._worker,
                        name=f"Worker{rank}",
                        args=(rank, child_pipe, parent_pipe))
      proc.daemon = True
      proc.start()
      child_pipe.close()
      self.pipes.append(parent_pipe)
      self.procs.append(proc)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    assert self.close()

  def __len__(self):
    return self.num_workers * self.models_per_worker
  
  def _build_env(self):
    raise NotImplementedError

  def _build_model(self):
    raise NotImplementedError

  def _evaluate(self, env, model):
    raise NotImplementedError

  def _worker(self, rank, pipe, parent_pipe):
    parent_pipe.close()
    envs = [self._build_env() for _ in range(self.models_per_worker)]
    models = [self._build_model() for _ in range(self.models_per_worker)]
    while True:
      command, data = pipe.recv()
      if command == "evaluate":
        seeds, solutions = data
        fitness = []
        for env, model, seed, solution in zip(envs, models, seeds, solutions):
          env.seed(seed)
          model.set_params(solution, precision=self.precision)
          fitness.append(self._evaluate(env, model))
        fitness = np.array(fitness)
        pipe.send((fitness, True))
      elif command == "close":
        pipe.send((None, True))
        return True
      else:
        raise NotImplementedError

  def evaluate(self, seeds, solutions):
    for i, pipe in enumerate(self.pipes):
      start = i * self.models_per_worker
      end = start + self.models_per_worker
      pipe.send(("evaluate", (seeds[start:end], solutions[start:end])))
    fitness, success = zip(*[pipe.recv() for pipe in self.pipes])
    return np.concatenate(fitness), all(success)

  def close(self):
    for pipe in self.pipes:
      pipe.send(("close", None))
    _, success = zip(*[pipe.recv() for pipe in self.pipes])
    return all(success)

class OpenES:
  def __init__(self, mu_init,
               sigma_init=0.1,
               sigma_decay=0.0001,
               sigma_limit=0.01,
               antithetic=True,
               stepsize=0.01,
               momentum=0.9):
    self.mu = mu_init
    self.sigma = sigma_init
    self.sigma_decay = sigma_decay
    self.sigma_limit = sigma_limit
    self.antithetic = antithetic
    self.stepsize = stepsize
    self.momentum = momentum
    self.v = np.zeros_like(self.mu)
    self.epsilon = None

  def sample(self, popsize):
    if self.antithetic:
      assert popsize % 2 == 0
      eps_split = np.random.randn(popsize // 2, self.mu.size)
      self.epsilon = np.concatenate([-eps_split, eps_split], axis=0)
    else:
      self.epsilon = np.random.randn(popsize, self.mu.size)
    return self.mu + self.sigma * self.epsilon

  def step(self, fitness):
    assert self.epsilon is not None
    assert len(fitness.shape) == 1
    popsize = fitness.shape[0]
    # shape fitness values as normalized ranks (higher the better)
    rank = np.empty_like(fitness, dtype=np.long)
    rank[np.argsort(fitness)] = np.arange(popsize)
    rank = rank.astype(np.float) / (popsize - 1) - 0.5
    fitness = (rank - np.mean(rank)) / np.std(rank)
    grad = 1 / (popsize * self.sigma) * (self.epsilon.T @ fitness)
    self.sigma = max(self.sigma * (1 - self.sigma_decay), self.sigma_limit)
    # gradient ascent with momentum
    self.v = self.momentum * self.v + (1 - self.momentum) * grad
    self.mu += self.stepsize * self.v
