import argparse
import numpy as np
from evomini.ga import SimpleGA
from evomini.nn import Module, Linear, LSTM
from evomini.eval import Evaluator
from cartpole_swingup import CartPoleSwingUpEnv

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num-workers", type=int, default=16)
parser.add_argument("--models-per-worker", type=int, default=16)
parser.add_argument("--num-gen", type=int, default=1000)
parser.add_argument("--num-evals", type=int, default=1)
parser.add_argument("--num-topk-evals", type=int, default=5)
parser.add_argument("--precision", type=int, default=4)
parser.add_argument("--sigma", type=float, default=0.1)
parser.add_argument("--mut_sigma", type=float, default=0.01)
args = parser.parse_args()

np.random.seed(args.seed)

class Model(Module):
  # small world model agent
  def __init__(self, obs_size, action_size, hidden_size):
    super().__init__()
    self.obs_size = obs_size
    self.action_size = action_size
    self.hidden_size = hidden_size
    self.register_module("C", Linear(obs_size + hidden_size, action_size))
    self.register_module("M", LSTM(obs_size + action_size, hidden_size))

  def __call__(self, *args, module="C"):
    if module == "C":
      obs, h = args
      x = np.concatenate([obs, h])
      action = self.C(x)
      return action
    if module == "M":
      obs, action = args
      x = np.concatenate([obs, action])
      h = self.M(x)
      return h

class CartPoleSwingUpEvaluator(Evaluator):
  def _build_env(self):
    return CartPoleSwingUpEnv()
  
  def _build_model(self):
    return Model(5, 1, 16)

  def _evaluate_once(self, env, model):
    obs = env.reset()
    h = model.M.reset()
    rewards = 0
    done = False
    while not done:
      action = model(obs, h, module="C")
      obs, reward, done, _ = env.step(action)
      h = model(obs, action, module="M")
      rewards += reward
    return rewards

env = CartPoleSwingUpEnv()
num_params = len(Model(5, 1, 16))
ga = SimpleGA(num_params, sigma=args.sigma, mut_sigma=args.mut_sigma)
global_best_fitness = -np.inf

with CartPoleSwingUpEvaluator(args.num_workers,
                              args.models_per_worker,
                              args.precision) as evaluator:
  popsize = len(evaluator)
  seeds, solutions = ga.sample(popsize)

  for gen in range(args.num_gen):
    fitness, success = evaluator.evaluate(seeds, solutions, args.num_evals)
    assert success, f"evaluation failed at generation {gen}"

    topk_seeds, topk_solutions = ga.set_elite_candidates(fitness)
    topk_fitness, success = evaluator.evaluate(topk_seeds, topk_solutions, args.num_topk_evals)
    assert success, f"topk evaluation failed at generation {gen}"
    elite_solution = ga.set_elite(topk_fitness)

    seeds, solutions = ga.reproduce()

    elite_fitness = np.max(topk_fitness)
    if elite_fitness > global_best_fitness:
      print(f"improvement detected: {global_best_fitness} -> {elite_fitness}")
      np.save("model_final.npy", elite_solution)
      global_best_fitness = elite_fitness

    stats = {
      "gen": gen,
      "pop_fitness_mean": np.mean(fitness),
      "pop_fitness_std": np.std(fitness),
      "pop_fitness_max": np.max(fitness),
      "pop_fitness_min": np.min(fitness),
      "topk_fitness_mean": np.mean(topk_fitness),
      "topk_fitness_std": np.std(topk_fitness),
      "elite_fitness": elite_fitness,
    }
    print(stats)
