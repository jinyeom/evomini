import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / np.sum(e_x, axis=0)

class Module:
  def __init__(self):
    self._children = {}
    self._params = []

  @property
  def params(self):
    return np.concatenate([np.ravel(p) for p in self._params])

  def __len__(self):
    return sum(p.size for p in self._params)

  def register_module(self, name, module):
    assert isinstance(module, Module)
    setattr(self, name, module)
    self._children.update({name: module})
    self._params.extend(module._params)

  def register_param(self, name, shape):
    param = np.empty(shape)
    setattr(self, name, param)
    self._params.append(param)

  def set_params(self, params, precision=None):
    assert params.size == len(self)
    if precision is not None:
      params = np.round(params * 10 ** precision) / 10 ** precision
    for dst in self._params:
      src, params = params[:dst.size], params[dst.size:]
      np.copyto(dst, np.array(src).reshape(dst.shape))

class Linear(Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.register_param("W", (input_size, output_size))
    self.register_param("b", (output_size,))

  def __call__(self, x):
    return x @ self.W + self.b

class RNN(Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.register_param("W", (input_size + hidden_size, hidden_size))
    self.register_param("b", (hidden_size,))
    self.reset()

  def reset(self):
    self.h = np.zeros(self.hidden_size)
    return np.array(self.h)

  def __call__(self, x):
    xh = np.concatenate([x, self.h])
    out = xh @ self.W + self.b
    out = (out - np.mean(out)) / np.std(out)
    self.h = np.tanh(out)
    return np.array(self.h)

class LSTM(Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.register_param("W", (input_size + hidden_size, 4 * hidden_size))
    self.register_param("b", (4 * hidden_size,))
    self.reset()

  def reset(self):
    self.h = np.zeros(self.hidden_size)
    self.c = np.zeros(self.hidden_size)
    return np.array(self.h)

  def __call__(self, x):
    xh = np.concatenate([x, self.h])
    out = xh @ self.W + self.b
    out = (out - np.mean(out)) / np.std(out)
    out = np.split(out, 4)
    f = sigmoid(out[0])
    i = sigmoid(out[1])
    o = sigmoid(out[2])
    c = np.tanh(out[3])
    self.c = f * self.c + i * c
    self.h = o * np.tanh(self.c)
    return np.array(self.h)
