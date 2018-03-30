"""A set of utilities designed to assist with handling and debugging NaNs in
pytorch.

* `isnan()` is an element-wise NaN check for pytorch Tensors.
* `hasnan()` determines whether or not a pytorch Tensor has any NaNs.
* `findnan()` will attempt to find NaNs in arbitrarily nested Python objects.

This module also offers a wrapped version of the entire `torch` module which
checks for NaNs in all inputs and outputs from all `torch` functions. For
example,

```
In [1]: from kindling.nan_police import torch

In [2]: x = torch.ones(2) / 0 * 0

In [3]: x
Out[3]:

nan
nan
[torch.FloatTensor of size 2]

In [4]: torch.sum(x)
---------------------------------------------------------------------------
Exception                                 Traceback (most recent call last)
<ipython-input-4-e7f45fec8fb4> in <module>()
----> 1 torch.sum(x)

~/Development/kindling/kindling/nan_police.py in __call__(self, *args, **kwargs)
    147         if argnan_path == []:
    148           raise Exception(
--> 149             f'Found a NaN at positional argument {i + 1} (of {len(args)}) when '
    150             f'calling `{path}`!'
    151           )

Exception: Found a NaN at positional argument 1 (of 1) when calling `torch.sum`!
```

Just import `from kindling.nan_police import torch` in all of your modules when
you'd like to test for NaNs! Note that this won't inject itself into all
references to torch, only those where you've imported this wrapped version of
pytorch. That means that it won't affect the behavior of torch in third-party
modules for example.
"""

import math
import numbers

import numpy as np
import torch as realtorch


def isnan(tensor):
  # Gross: https://github.com/pytorch/pytorch/issues/4767
  return (tensor != tensor)

def hasnan(tensor):
  return isnan(tensor).any()

def findnan(obj):
  """Find a NaN in the haystack of `obj` and return the path to finding it.
  Returns None if no NaNs could be found. Works with Python numbers, lists,
  tuples, Pytorch Tensors, Variables, and numpy arrays."""
  # Python numbers
  if isinstance(obj, numbers.Number):
    return ([] if math.isnan(obj) else None)

  # Pytorch Tensors
  elif isinstance(obj, realtorch.Tensor):
    return ([] if hasnan(obj) else None)

  # Pytorch Variables
  elif isinstance(obj, realtorch.autograd.Variable):
    if hasnan(obj.data):
      return [('Variable.data',)]
    else:
      gradnan_path = findnan(obj.grad)
      if gradnan_path == None:
        return None
      else:
        return [('Variable.grad',)] + gradnan_path

  # numpy array (See https://stackoverflow.com/questions/36783921/valueerror-when-checking-if-variable-is-none-or-numpy-array)
  elif isinstance(obj, np.ndarray):
    return ([] if np.isnan(obj).any() else None)

  # Python list
  elif isinstance(obj, list):
    for ix, elem in enumerate(obj):
      elemnan_path = findnan(elem)
      if elemnan_path != None:
        return [('list[]', ix)] + elemnan_path

    # Haven't found any NaNs yet, we're done
    return None

  # Python tuple
  elif isinstance(obj, tuple):
    for ix, elem in enumerate(obj):
      elemnan_path = findnan(elem)
      if elemnan_path != None:
        return [('tuple[]', ix)] + elemnan_path

    # Haven't found any NaNs yet, we're done
    return None

  # Python dict
  elif isinstance(obj, dict):
    for k, v in obj.items():
      elemnan_path = findnan(v)
      if elemnan_path != None:
        return [('dict[]', k)] + elemnan_path

    # Haven't found any NaNs yet, we're done
    return None

  # Don't know what we're looking at, assume it doesn't have NaNs.
  else:
    return None

def path_to_string(path):
  """Convert a path found with `findnan` to a pseudo-code selector."""
  if len(path) == 0:
    return ''
  else:
    first = path[0]
    rest = path[1:]
    typ = first[0]
    if typ == 'Variable.data':
      return '.data' + path_to_string(rest)
    elif typ == 'Variable.grad':
      return '.grad' + path_to_string(rest)
    elif (typ == 'list[]') or (typ == 'tuple[]') or (typ == 'dict[]'):
      return f'[{first[1]}]' + path_to_string(rest)
    else:
      raise Exception(f'Unrecognized path type, {typ}')

class Mock(object):
  """Emulate a module or function and wrap all calls with NaN checking. Throws
  an Exception when any NaNs are found."""
  def __init__(self, obj, path):
    self._obj = obj
    self._path = path

  def __call__(self, *args, **kwargs):
    path = '.'.join(self._path)

    # check args
    for i, arg in enumerate(args):
      argnan_path = findnan(arg)
      if argnan_path != None:
        if argnan_path == []:
          raise Exception(
            f'Found a NaN at positional argument {i + 1} (of {len(args)}) when '
            f'calling `{path}`!'
          )
        else:
          raise Exception(
            f'Found NaN in positional argument {i + 1} (of {len(args)}) when '
            f'calling `{path}`! Specifically at '
            f'`<arg>{path_to_string(argnan_path)}`.'
          )

    # check kwargs
    for k, v in kwargs.items():
      vnan_path = findnan(v)
      if vnan_path != None:
        if vnan_path == []:
          raise Exception(
            f'Found a NaN at keyword argument `{k}` when calling `{path}`!'
          )
        else:
          raise Exception(
            f'Found NaN in keyword argument `{k}` when calling `{path}`! '
            f'Specifically at `<{k}>{path_to_string(vnan_path)}`.'
          )

    result = self._obj(*args, **kwargs)

    # check result for NaNs
    resultnan_path = findnan(result)
    if resultnan_path != None:
      if resultnan_path == []:
        raise Exception(f'Found NaN in output from `{path}`!')
      else:
        raise Exception(
          f'Found NaN in output from `{path}`! Specifically at '
          f'`<out>{path_to_string(resultnan_path)}`.'
        )

    return result

  def __getattr__(self, name):
    return Mock(getattr(self._obj, name), self._path + [name])

  def __dir__(self):
    return self._obj.__dir__()

torch = Mock(realtorch, ['torch'])
