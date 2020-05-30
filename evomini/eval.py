import multiprocessing as mp

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

  def _evaluate_once(self, env, model):
    raise NotImplementedError

  def _evaluate(self, env, model):
    fitness = [self._evaluate_once(env, model) for _ in range(self.num_evals)]
    return np.mean(fitness)

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
          env.seed(int(seed))
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
