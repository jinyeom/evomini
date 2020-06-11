import numpy as np

def tanh(x):
  return np.tanh(x)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return x * (x > 0)

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

  def __call__(self):
    raise NotImplementedError

class Tanh(Module):
  def __call__(self, x):
    return tanh(x)

class Sigmoid(Module):
  def __call__(self, x):
    return sigmoid(x)

class ReLU(Module):
  def __call__(self, x):
    return relu(x)

class Softmax(Module):
  def __call__(self, x):
    return softmax(x)

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
    self._h = np.zeros(self.hidden_size, dtype=np.float16)
    return np.array(self._h)

  def __call__(self, x):
    xh = np.concatenate([x, self._h])
    out = xh @ self.W + self.b
    out = (out - np.mean(out)) / np.std(out)
    self._h = np.tanh(out)
    return np.array(self._h)

class LSTM(Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.register_param("W", (input_size + hidden_size, 4 * hidden_size))
    self.register_param("b", (4 * hidden_size,))
    self.reset()

  def reset(self):
    self._h = np.zeros(self.hidden_size, dtype=np.float16)
    self._c = np.zeros(self.hidden_size, dtype=np.float16)
    return np.array(self._h)

  def __call__(self, x):
    xh = np.concatenate([x, self._h])
    out = xh @ self.W + self.b
    out = (out - np.mean(out)) / np.std(out)
    out = np.split(out, 4)
    f = sigmoid(out[0])
    i = sigmoid(out[1])
    o = sigmoid(out[2])
    c = np.tanh(out[3])
    self._c = f * self._c + i * c
    self._h = o * np.tanh(self._c)
    return np.array(self._h)

class NPRNN(Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.register_param("W", (input_size, hidden_size))
    self.register_param("U", (hidden_size, hidden_size))
    self.register_param("A", (hidden_size, hidden_size))
    self.register_param("b", (hidden_size,))
    self.register_module("modulator", Linear(hidden_size, 1))
    self.register_module("modfanout", Linear(1, hidden_size))
    self.reset()

  def reset(self):
    self._h = np.zeros(self.hidden_size, dtype=np.float16)
    self._hebb = np.zeros((self.hidden_size, self.hidden_size), dtype=np.float16)
    return np.array(self._h)

  def __call__(self, x):
    h0 = self._h
    z = x @ self.W + self.b
    U = self.U + self.A * self.hebb
    out = z + self._h @ U
    out = (out - np.mean(out)) / np.std(out)
    h1 = np.tanh(out)
    M = np.tanh(self.modulator(h1))
    M = self.modfanout(M)[:, np.newaxis]
    self._hebb += M * np.outer(h0, h1)
    self._h = h1
    return np.array(self._h)

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
    self._memory = np.zeros((1, self.mem_size), dtype=np.float16)
    self._head = 0
    self._M_h = self._memory[self._head]
    return self.controller.reset()

  def _write(self, h):
    a = self.write_vec(h)
    w = sigmoid(self.write_interp(h))
    M_h = (1 - w) * self._M_h + w * a
    self._memory[self._head] = M_h
    return a, w

  def _content_jump(self, h, a):
    j = sigmoid(self.content_jump(h))
    if j > 0.5:
      dist = np.sqrt(np.sum((self._memory - a) ** 2, axis=1))
      self._head = int(np.argmin(dist))
    return j

  def _shift(self, h):
    s = softmax(self.shift(h))
    self._head += np.argmax(s) - 1
    if self._head < 0:
      mem_ext = np.zeros((1, self.mem_size,))
      self._memory = np.concatenate([mem_ext, self._memory], axis=0)
      self._head = 0
    elif self._head >= self._memory.shape[0]:
      mem_ext = np.zeros((1, self.mem_size,))
      self._memory = np.concatenate([self._memory, mem_ext], axis=0)
    return s

  def _read(self):
    self._M_h = self._memory[self._head]
    return np.array(self._M_h)

  def __call__(self, x):
    x = np.concatenate([x, self._M_h])
    h = self.controller(x)
    a, w = self._write(h)
    j = self._content_jump(h, a)
    s = self._shift(h)
    M_h = self._read()
    return h
