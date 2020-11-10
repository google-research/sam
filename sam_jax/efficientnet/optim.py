# Copyright 2020 The Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EfficientNet is trained with TF1 RMSProp, which is reproduced here.

This version has two difference with the FLAX RMSProp version:
  * It supports momentum, which the FLAX version does not.
  * The moving average of the squared gradient is initialized to 1 as in TF1,
    instead of 0 as in everywhere else.
"""

from typing import Any, Optional, Tuple

from flax import struct
from flax.optim.base import OptimizerDef
import jax
import jax.numpy as jnp
import numpy as onp


# pytype:disable=wrong-arg-count
@struct.dataclass
class _RMSPropHyperParams:
  """RMSProp hyper parameters."""

  learning_rate: float
  beta: float
  beta2: float
  eps: float


@struct.dataclass
class _RMSPropParamState:
  """RMSProp parameter state."""

  v: onp.ndarray
  momentum: onp.ndarray


class RMSProp(OptimizerDef):
  """RMSProp optimizer."""

  def __init__(self, learning_rate: Optional[float] = None, beta: float = 0.0,
               beta2: float = 0.9, eps: float = 1e-8):
    """Instantiates the RMSProp optimizer.

    Args:
      learning_rate: The step size used to update the parameters.
      beta: Momentum.
      beta2: The coefficient used for the moving average of the
        gradient magnitude (default: 0.9).
      eps: The term added to the gradient magnitude estimate for
        numerical stability.
    """
    hyper_params = _RMSPropHyperParams(learning_rate, beta, beta2, eps)
    super().__init__(hyper_params)

  def init_param_state(self, param: jnp.ndarray) -> _RMSPropParamState:
    """Initializes parameter state. See base class."""
    return _RMSPropParamState(
        jnp.ones_like(param), jnp.zeros_like(param))

  def apply_param_gradient(
      self, step: jnp.ndarray,
      hyper_params: _RMSPropHyperParams,
      param: jnp.ndarray,
      state: _RMSPropParamState,
      grad: jnp.ndarray) -> Tuple[jnp.ndarray, _RMSPropParamState]:
    """Applies per-parameter gradients. See base class."""
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    new_v = hyper_params.beta2 * state.v + (
        1.0 - hyper_params.beta2) * jnp.square(grad)
    grad = grad / jnp.sqrt(new_v + hyper_params.eps)
    new_momentum = hyper_params.beta * state.momentum + grad
    new_param = param - hyper_params.learning_rate * new_momentum
    new_state = _RMSPropParamState(new_v, new_momentum)

    return new_param, new_state


# pytype:disable=attribute-error
@struct.dataclass
class ExponentialMovingAverage:
  """Exponential Moving Average as implemented in Tensorflow."""

  # Moving average of the parameters.
  param_ema: Any
  # Decay to use for the update (typical values are 0.999, 0.9999, etc...).
  decay: float
  # For how many steps we should just keep the new parameters instead of an
  # average (useful if we don't want the initial weights to be included in the
  # average).
  warmup_steps: int

  def update_moving_average(self, new_target: Any,
                            step: jnp.ndarray) -> Any:
    """Updates the moving average of the target.

    Args:
      new_target: New values of the target (example: weights of a network
        after gradient step).
      step: Current step (used only for warmup).

    Returns:
      The updated ExponentialMovingAverage.
    """
    factor = jnp.float32(step >= self.warmup_steps)
    delta = step - self.warmup_steps
    decay = jnp.minimum(self.decay, (1. + delta) / (10. + delta))
    decay *= factor
    weight_ema = jax.tree_multimap(
        lambda a, b: (1 - decay) * a + decay * b, new_target, self.param_ema)
    return self.replace(param_ema=weight_ema)
# pytype:enable=attribute-error
