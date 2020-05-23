import argparse
import numpy as np
from evomini.es import Evaluator, OpenaiES
from evomini.nn import Module, Linear, LSTM
from cartpole_swingup import CartPoleSwingUpEnv

parser = argparse.ArgumentParser()
parser.add_argument("--artifact-path", type=str, default="artifacts/")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--desc", type=str, default=None)
parser.add_argument("--num-workers", type=int, default=16)
parser.add_argument("--models-per-worker", type=int, default=16)
parser.add_argument("--num-gen", type=int, default=1000)
parser.add_argument("--num-evals", type=int, default=5)
parser.add_argument("--precision", type=int, default=4)
parser.add_argument("--sigma-init", type=float, default=0.1)
parser.add_argument("--sigma-decay", type=float, default=0.0001)
parser.add_argument("--sigma-limit", type=float, default=0.01)
parser.add_argument("--antithetic", type=bool, default=True)
parser.add_argument("--stepsize", type=float, default=0.03)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--eval-interval", type=int, default=50)
args = parser.parse_args()

np.random.seed(args.seed)

class Model(Module):
  def __init__(self, obs_size, action_size, hidden_size):
    super().__init__()
    self.obs_size = obs_size
    self.action_size = action_size
    self.hidden_size = hidden_size
    self.register_module("rnn", LSTM(obs_size, hidden_size))
    self.register_module("controller", Linear(obs_size + hidden_size, action_size))

  def __call__(self, obs):
    h = self.rnn(obs)
    x = np.concatenate([obs, h])
    action = self.controller(x)
    return action

class CartPoleSwingUpEvaluator(Evaluator):
  def _build_env(self):
    return CartPoleSwingUpEnv()
  
  def _build_model(self):
    return Model(5, 1, 16)

  def _evaluate_once(self, env, model):
    obs = env.reset()
    model.rnn.reset()
    rewards = 0
    done = False
    while not done:
      action = model(obs)
      obs, reward, done, _ = env.step(action)
      rewards += reward
    return reward

env = CartPoleSwingUpEnv()
model = Model(5, 1, 16)
mu_init = np.zeros(len(model))
es = OpenaiES(mu_init,
              sigma_init=args.sigma_init,
              sigma_decay=args.sigma_decay,
              sigma_limit=args.sigma_limit,
              antithetic=args.antithetic,
              stepsize=args.stepsize)
global_best_fitness = -np.inf

with CartPoleSwingUpEvaluator(args.num_workers,
                              args.models_per_worker,
                              args.num_evals,
                              args.precision) as evaluator:
  popsize = len(evaluator)
  for gen in range(args.num_gen):
    seeds = np.random.randint(2 ** 31 - 1, size=popsize).tolist()
    solutions = es.sample(popsize)
    fitness, success = evaluator.evaluate(seeds, solutions)
    assert success, f"evaluation failed at generation {gen}"
    es.step(fitness)

    best_fitness = np.max(fitness)
    if best_fitness > global_best_fitness:
      print(f"improvement detected: {global_best_fitness} -> {best_fitness}")
      best = solutions[np.argmax(fitness)]
      np.save("model_final.npy", best)
      global_best_fitness = best_fitness

    stats = {
      "gen": gen,
      "fitness_mean": np.mean(fitness),
      "fitness_std": np.std(fitness),
      "fitness_max": np.max(fitness),
      "fitness_min": np.min(fitness),
    }
    print(stats)
