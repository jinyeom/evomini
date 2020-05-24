import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / np.sum(e_x, axis=0)

class Module:
  def __init__(self):
    self._children = []
    self._params = []

  @property
  def params(self):
    return np.concatenate([np.ravel(p) for p in self._params])

  def __len__(self):
    return sum(p.size for p in self._params)

  def register_module(self, name, module):
    assert isinstance(module, Module)
    setattr(self, name, module)
    self._children.append((name, module))
    self._params.extend(module._params)

  def register_param(self, name, shape):
    param = np.empty(shape, dtype=np.float16)
    setattr(self, name, param)
    self._params.append(param)

  def set_params(self, params, precision=None):
    assert params.size == len(self)
    if precision is not None:
      params = np.round(params * 10 ** precision) / 10 ** precision
    for dst in self._params:
      src, params = params[:dst.size], params[dst.size:]
      src = np.array(src, dtype=np.float16).reshape(dst.shape)
      np.copyto(dst, src)

  def reset(self):
    return None

class Stack(Module):
  def __init__(self, **kwargs):
    super().__init__()
    for name, module in kwargs.items():
      self.register_module(name, module)

  def reset(self):
    hiddens = {}
    for name, module in self._children:
      hiddens[name] = module.reset()
    return hiddens

  def __call__(self, x):
    for _, module in self._children:
      x = module(x)
    return x

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
    self.h = np.zeros(self.hidden_size, dtype=np.float16)
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
    self.h = np.zeros(self.hidden_size, dtype=np.float16)
    self.c = np.zeros(self.hidden_size, dtype=np.float16)
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

class ENTM(Module):
  def __init__(self, input_size, hidden_size, mem_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.mem_size = mem_size
    self.register_module("controller", LSTM(input_size + mem_size, hidden_size))
    self.register_module("write_vec", Linear(hidden_size, mem_size))
    self.register_module("write_interp", Linear(hidden_size, 1))
    self.register_module("content_jump", Linear(hidden_size, 1))
    self.register_module("shift", Linear(hidden_size, 3))
    self.reset()

  def reset(self):
    self.memory = np.zeros((1, self.mem_size), dtype=np.float16)
    self.head = 0
    self.M_h = self.memory[self.head]
    return self.controller.reset()

  def _write(self, h):
    a = self.write_vec(h)
    w = sigmoid(self.write_interp(h))
    M_h = (1 - w) * self.M_h + w * a
    self.memory[self.head] = M_h
    return a, w

  def _content_jump(self, h, a):
    j = sigmoid(self.content_jump(h))
    if j > 0.5:
      dist = np.sqrt(np.sum((self.memory - a) ** 2, axis=1))
      self.head = int(np.argmin(dist))
    return j

  def _shift(self, h):
    s = softmax(self.shift(h))
    self.head += np.argmax(s) - 1
    if self.head < 0:
      mem_ext = np.zeros((1, self.mem_size,))
      self.memory = np.concatenate([mem_ext, self.memory], axis=0)
      self.head = 0
    elif self.head >= self.memory.shape[0]:
      mem_ext = np.zeros((1, self.mem_size,))
      self.memory = np.concatenate([self.memory, mem_ext], axis=0)
    return s

  def _read(self):
    self.M_h = self.memory[self.head]
    return np.array(self.M_h)

  def __call__(self, x):
    x = np.concatenate([x, self.M_h])
    h = self.controller(x)
    a, w = self._write(h)
    j = self._content_jump(h, a)
    s = self._shift(h)
    M_h = self._read()
    return h
