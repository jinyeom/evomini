# evomini
`evomini` is a neuroevolution framework that consists of minimal implementations of neural network modules and evolutionary algorithms. The primary purpose of this project is to better undertand existing neuroevolution algorithms and help further develop novel algorithms.

### Modules
- [x] Basic modules (Linear, RNN)
- [x] LSTM (Long Short-Term Memory)
- [x] ENTM (Evolvable Neural Turing Machine)

### Algorithms
- [x] OpenAI-ES (simplified natural evolution strategy)
- [ ] ME-ES (MAP-Elites Evolution Strategy)

If you'd like to use this project for your research, please use the following bibtex to cite this repository:
```
@misc{evomini,
  author = {Yeom, Jin},
  title = {Minimal implementation of neuroevolution algorithms},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jinyeom/evomini}},
}
```

## Requirements
- NumPy
- _That's it!_

## How to use it
Using `evomini` consists of four steps:
1. Define agent architecture
2. Define environment
3. Implement `Evaluator`
4. Start evolution

A full implementation of this example can be found under [examples](https://github.com/jinyeom/evomini/tree/master/examples).

### 1. Define agent architecture
First, we must define your agent's architecture using `evomini.nn.Module` and other included modules, such as `Linear`, `RNN`, etc. In this example, we build small end-to-end world model agents ([[Ha and Schmidhuber, 2018](https://arxiv.org/abs/1803.10122)], [[Risi and Stanley, 2019](https://arxiv.org/abs/1906.08857)]).

```python
from evomini.nn import Module, Linear, LSTM

class Model(Module):
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
```

### 2. Define environment
Next, we define our environment. If you have worked with reinforcement learning, this should be a familiar step. In this example, we'll use `CartPoleSwingUpEnv`, which you can find under [examples/cartpole_swingup](https://github.com/jinyeom/evomini/tree/master/examples/cartpole_swingup).

```python
class CartPoleSwingUpEnv(gym.Env):
    ...
```

### 3. Implement `Evaluator`
Once our agent and environment are defined, we can build our parallel evaluator. All we need to do is implementing three of its methods: `_build_env()`, `_build_model()`, and `_evaluate_once`. Respectively, they build an instance of the environment, build an instance of the agent, and evaluate an agent in an environment to retrieve its fitness value.

```python
from evomini.es import Evaluator

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
```

### 4. Start evolution
Now that all the components are ready, we can put them together to start our evolution.

```python
# create an evolutionary algorithm instace
es = OpenaiES(np.zeros(len(Model(5, 1, 16))),
              sigma_init=sigma_init,
              sigma_decay=sigma_decay,
              sigma_limit=sigma_limit,
              antithetic=antithetic,
              stepsize=stepsize)

# start evolution
with CartPoleSwingUpEvaluator(num_workers, models_per_worker, num_evals, precision) as evaluator:
  for gen in range(num_gen):
    seeds, solutions = es.sample(len(evaluator))
    fitness, success = evaluator.evaluate(seeds, solutions)
    assert success, f"evaluation failed at generation {gen}"
    es.step(fitness)

    ...
```

## Related projects
- [estool](https://github.com/hardmaru/estool)

## Todo
- [ ] Separate optimizer from ES classes
