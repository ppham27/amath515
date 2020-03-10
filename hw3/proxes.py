"""Implementation of `prox_csimplex`."""
import numpy as np
from scipy.optimize import bisect

UW_ID = '1772371'
FIRST_NAME = 'Philip'
LAST_NAME = 'Pham'

# Prox of capped simplex
# -----------------------------------------------------------------------------
def prox_csimplex(z, k):
  """Prox of capped simplex argmin_x 1/2||x - z||^2 s.t. x in k-capped-simplex.

  Args:
    z: arraylike, reference point
    k: float, positive number between 0 and z.size, denote simplex cap

  Returns:
    arraylike, projection of z onto the k-capped simplex
  """
  # safe guard for k
  if k < 0 or k > z.size:
    raise ValueError(
      'k: k must be between 0 and dimension of the input. k = {}'.format(k))
  # 1. Construct the scalar dual object.
  def f(y):
    return np.sum(np.clip(z - y, 0, 1)) - k
  # 2. Use `bisect` to solve it.
  lower_bound, upper_bound = -1., 1.
  while f(lower_bound) < 0:
    lower_bound *= 2
  while f(upper_bound) >= 0:
    upper_bound *= 2
  y = bisect(f, lower_bound, upper_bound)
  # 3. Obtain primal variable from optimal dual solution and return it.
  return np.clip(z - y, 0, 1)
