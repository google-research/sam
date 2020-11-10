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

"""Defines Efficientnet model."""

import copy
import math
from typing import Any, Optional, Tuple, Union

from absl import flags
from absl import logging
import flax
from flax import nn
import jax
from jax import numpy as jnp
import tensorflow as tf


FLAGS = flags.FLAGS


def name_to_image_size(name: str) -> int:
  """Returns the expected image size for a given model.

  If the model is not a recognized efficientnet model, will default to the
  standard resolution of 224 (for Resnet, etc...).

  Args:
    name: Name of the efficientnet model (ex: efficientnet-b0).
  """
  image_sizes = {
      'efficientnet-b0': 224,
      'efficientnet-b1': 240,
      'efficientnet-b2': 260,
      'efficientnet-b3': 300,
      'efficientnet-b4': 380,
      'efficientnet-b5': 456,
      'efficientnet-b6': 528,
      'efficientnet-b7': 600,
      'efficientnet-b8': 672,
      'efficientnet-l2': 800,
      'efficientnet-l2-475': 475,
  }
  return image_sizes.get(name, 224)


# Relevant initializers. The original implementation uses fan_out Kaiming init.

conv_kernel_init_fn = jax.nn.initializers.variance_scaling(
    2.0, 'fan_out', 'truncated_normal')

dense_kernel_init_fn = jax.nn.initializers.variance_scaling(
    1 / 3.0, 'fan_out', 'uniform')


class DepthwiseConv(flax.nn.Module):
  """Depthwise convolution that matches tensorflow's conventions.

  In Tensorflow, the shapes of depthwise kernels don't match the shapes of a
  regular convolutional kernel of appropriate feature_group_count.
  It is safer to use this class instead of the regular Conv (easier port of
  tensorflow checkpoints, fan_out initialization of the previous layer will
  match the tensorflow behavior, etc...).
  """

  def apply(self,
            inputs: jnp.ndarray,
            features: int,
            kernel_size: Tuple[int, int],
            strides: bool = None,
            padding: str = 'SAME',
            input_dilation: int = None,
            kernel_dilation: int = None,
            bias: bool = True,
            dtype: jnp.dtype = jnp.float32,
            precision=None,
            kernel_init=flax.nn.initializers.lecun_normal(),
            bias_init=flax.nn.initializers.zeros) -> jnp.ndarray:
    """Applies a convolution to the inputs.

    Args:
      inputs: Input data with dimensions (batch, spatial_dims..., features).
      features: Number of convolution filters.
      kernel_size: Shape of the convolutional kernel.
      strides: A sequence of `n` integers, representing the inter-window
        strides.
      padding: Either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension.
      input_dilation: `None`, or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`.
        Convolution with input dilation `d` is equivalent to transposed
        convolution with stride `d`.
      kernel_dilation: `None`, or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel. Convolution with kernel dilation is also known as 'atrous
        convolution'.
      bias: Whether to add a bias to the output (default: True).
      dtype: The dtype of the computation (default: float32).
      precision: Numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: Initializer for the convolutional kernel.
      bias_init: Initializer for the bias.

    Returns:
      The convolved data.
    """

    inputs = jnp.asarray(inputs, dtype)
    in_features = inputs.shape[-1]

    if strides is None:
      strides = (1,) * (inputs.ndim - 2)

    kernel_shape = kernel_size + (features, 1)
    # Naming convention follows tensorflow.
    kernel = self.param('depthwise_kernel', kernel_shape, kernel_init)
    kernel = jnp.asarray(kernel, dtype)

    # Need to transpose to convert tensorflow-shaped kernel to lax-shaped kernel
    kernel = jnp.transpose(kernel, [0, 1, 3, 2])

    dimension_numbers = flax.nn.linear._conv_dimension_numbers(inputs.shape)  # pylint:disable=protected-access

    y = jax.lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=in_features,
        precision=precision)

    if bias:
      bias = self.param('bias', (features,), bias_init)
      bias = jnp.asarray(bias, dtype)
      y = y + bias
    return y


# pytype: disable=attribute-error
# pylint:disable=unused-argument
class BlockConfig(object):
  """Class that contains configuration parameters for a single block."""

  def __init__(self,
               input_filters: int = 0,
               output_filters: int = 0,
               kernel_size: int = 3,
               num_repeat: int = 1,
               expand_ratio: int = 1,
               strides: Tuple[int, int] = (1, 1),
               se_ratio: Optional[float] = None,
               id_skip: bool = True,
               fused_conv: bool = False,
               conv_type: str = 'depthwise'):
    for arg in locals().items():
      setattr(self, *arg)


class ModelConfig(object):
  """Class that contains configuration parameters for the model."""

  def __init__(
      self,
      width_coefficient: float = 1.0,
      depth_coefficient: float = 1.0,
      resolution: int = 224,
      dropout_rate: float = 0.2,
      blocks: Tuple[BlockConfig, ...] = (
          # (input_filters, output_filters, kernel_size, num_repeat,
          #  expand_ratio, strides, se_ratio)
          # pylint: disable=bad-whitespace
          BlockConfig(32, 16, 3, 1, 1, (1, 1), 0.25),
          BlockConfig(16, 24, 3, 2, 6, (2, 2), 0.25),
          BlockConfig(24, 40, 5, 2, 6, (2, 2), 0.25),
          BlockConfig(40, 80, 3, 3, 6, (2, 2), 0.25),
          BlockConfig(80, 112, 5, 3, 6, (1, 1), 0.25),
          BlockConfig(112, 192, 5, 4, 6, (2, 2), 0.25),
          BlockConfig(192, 320, 3, 1, 6, (1, 1), 0.25),
          # pylint: enable=bad-whitespace
      ),
      stem_base_filters: int = 32,
      top_base_filters: int = 1280,
      activation: str = 'swish',
      batch_norm: str = 'default',
      bn_momentum: float = 0.99,
      bn_epsilon: float = 1e-3,
      # While the original implementation used a weight decay of 1e-5,
      # tf.nn.l2_loss divides it by 2, so we halve this to compensate in Keras
      weight_decay: float = 5e-6,
      drop_connect_rate: float = 0.2,
      depth_divisor: int = 8,
      min_depth: Optional[int] = None,
      use_se: bool = True,
      input_channels: int = 3,
      model_name: str = 'efficientnet',
      rescale_input: bool = True,
      data_format: str = 'channels_last',
      dtype: str = 'float32'):
    """Default Config for Efficientnet-B0."""
    for arg in locals().items():
      setattr(self, *arg)
# pylint:enable=unused-argument


MODEL_CONFIGS = {
    # (width, depth, resolution, dropout)
    'efficientnet-b0': ModelConfig(1.0, 1.0, 224, 0.2),
    'efficientnet-b1': ModelConfig(1.0, 1.1, 240, 0.2),
    'efficientnet-b2': ModelConfig(1.1, 1.2, 260, 0.3),
    'efficientnet-b3': ModelConfig(1.2, 1.4, 300, 0.3),
    'efficientnet-b4': ModelConfig(1.4, 1.8, 380, 0.4),
    'efficientnet-b5': ModelConfig(1.6, 2.2, 456, 0.4),
    'efficientnet-b6': ModelConfig(1.8, 2.6, 528, 0.5),
    'efficientnet-b7': ModelConfig(2.0, 3.1, 600, 0.5),
    'efficientnet-b8': ModelConfig(2.2, 3.6, 672, 0.5),
    'efficientnet-l2': ModelConfig(4.3, 5.3, 800, 0.5),
    'efficientnet-l2-475': ModelConfig(4.3, 5.3, 475, 0.5),
}


def round_filters(filters: int,
                  config: ModelConfig) -> int:
  """Returns rounded number of filters based on width coefficient."""
  width_coefficient = config.width_coefficient
  min_depth = config.min_depth
  divisor = config.depth_divisor
  orig_filters = filters

  if not width_coefficient:
    return filters

  filters *= width_coefficient
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  logging.info('round_filter input=%s output=%s', orig_filters, new_filters)
  return int(new_filters)


def round_repeats(repeats: int, depth_coefficient: float) -> int:
  """Returns rounded number of repeats based on depth coefficient."""
  return int(math.ceil(depth_coefficient * repeats))


def conv2d(inputs: tf.Tensor,
           conv_filters: Optional[int],
           config: ModelConfig,
           kernel_size: Union[int, Tuple[int, int]] = (1, 1),
           strides: Tuple[int, int] = (1, 1),
           use_batch_norm: bool = True,
           use_bias: bool = False,
           activation: Any = None,
           depthwise: bool = False,
           train: bool = True,
           conv_name: str = None,
           bn_name: str = None) -> jnp.ndarray:
  """Convolutional layer with possibly batch norm and activation.

  Args:
    inputs: Input data with dimensions (batch, spatial_dims..., features).
    conv_filters: Number of convolution filters.
    config: Configuration for the model.
    kernel_size: Size of the kernel, as a tuple of int.
    strides: Strides for the convolution, as a tuple of int.
    use_batch_norm: Whether batch norm should be applied to the output.
    use_bias: Whether we should add bias to the output of the first convolution.
    activation: Name of the activation function to use.
    depthwise: If true, will use depthwise convolutions.
    train: Whether the model should behave in training or inference mode.
    conv_name: Name to give to the convolution layer.
    bn_name: Name to give to the batch norm layer.

  Returns:
    The output of the convolutional layer.
  """
  conv_fn = DepthwiseConv if depthwise else flax.nn.Conv
  kernel_size = ((kernel_size, kernel_size)
                 if isinstance(kernel_size, int) else tuple(kernel_size))
  conv_name = conv_name if conv_name else 'conv2d'
  bn_name = bn_name if bn_name else 'batch_normalization'

  x = conv_fn(
      inputs,
      conv_filters,
      kernel_size,
      tuple(strides),
      padding='SAME',
      bias=use_bias,
      kernel_init=conv_kernel_init_fn,
      name=conv_name)

  if use_batch_norm:
    x = nn.BatchNorm(
        x,
        use_running_average=not train or FLAGS.from_pretrained_checkpoint,
        momentum=config.bn_momentum,
        epsilon=config.bn_epsilon,
        name=bn_name,
        axis_name='batch')

  if activation is not None:
    x = getattr(flax.nn.activation, activation.lower())(x)
  return x


def stochastic_depth(inputs: jnp.ndarray,
                     survival_probability: float,
                     deterministic: bool = False,
                     rng: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Applies stochastic depth.

  Args:
    inputs: The inputs that should be randomly masked.
    survival_probability: 1 - the probablity of masking out a value.
    deterministic: If false the inputs are scaled by `1 / (1 - rate)` and
      masked, whereas if true, no mask is applied and the inputs are returned as
      is.
    rng: An optional `jax.random.PRNGKey`. By default `nn.make_rng()` will
      be used.

  Returns:
    The masked inputs.
  """
  if survival_probability == 1.0 or deterministic:
    return inputs

  if rng is None:
    rng = flax.nn.make_rng()
  mask_shape = [inputs.shape[0]]+ [1 for _ in inputs.shape[1:]]
  mask = jax.random.bernoulli(rng, p=survival_probability, shape=mask_shape)
  mask = jnp.tile(mask, [1] + list(inputs.shape[1:]))
  return jax.lax.select(mask, inputs / survival_probability,
                        jnp.zeros_like(inputs))


class SqueezeExcite(flax.nn.Module):
  """SqueezeExite block (see paper for more details.)"""

  def apply(self,
            x: jnp.ndarray,
            filters: int,
            block: BlockConfig,
            config: ModelConfig,
            train: bool) -> jnp.ndarray:
    """Applies a convolution to the inputs.

    Args:
      x: Input data with dimensions (batch, spatial_dims..., features).
      filters: Number of convolution filters.
      block: Configuration for this block.
      config: Configuration for the model.
      train: Whether the model is in training or inference mode.

    Returns:
      The output of the squeeze excite block.
    """
    conv_index = 0
    num_reduced_filters = max(1, int(block.input_filters * block.se_ratio))

    se = flax.nn.avg_pool(x, x.shape[1:3])
    se = conv2d(
        se,
        num_reduced_filters,
        config,
        use_bias=True,
        use_batch_norm=False,
        activation=config.activation,
        conv_name='reduce_conv2d_' + str(conv_index),
        train=train)
    conv_index += 1

    se = conv2d(
        se,
        filters,
        config,
        use_bias=True,
        use_batch_norm=False,
        activation='sigmoid',
        conv_name='expand_conv2d_' + str(conv_index),
        train=train)
    conv_index += 1
    x = x * se
    return x


class MBConvBlock(flax.nn.Module):
  """Main building component of Efficientnet."""

  def apply(self,
            inputs: jnp.ndarray,
            block: BlockConfig,
            config: ModelConfig,
            train: bool = False) -> jnp.ndarray:
    """Mobile Inverted Residual Bottleneck.

    Args:
      inputs: Input to the block.
      block: BlockConfig, arguments to create a Block.
      config: ModelConfig, a set of model parameters.
      train: Whether we are training or predicting.

    Returns:
      The output of the block.
    """
    use_se = config.use_se
    activation = config.activation
    drop_connect_rate = config.drop_connect_rate
    use_depthwise = block.conv_type != 'no_depthwise'

    filters = block.input_filters * block.expand_ratio

    x = inputs
    bn_index = 0
    conv_index = 0

    if block.fused_conv:
      # If we use fused mbconv, skip expansion and use regular conv.
      x = conv2d(
          x,
          filters,
          config,
          kernel_size=block.kernel_size,
          strides=block.strides,
          activation=activation,
          conv_name='fused_conv2d_' + str(conv_index),
          bn_name='batch_normalization_' + str(bn_index),
          train=train)
      bn_index += 1
      conv_index += 1
    else:
      if block.expand_ratio != 1:
        # Expansion phase
        kernel_size = (1, 1) if use_depthwise else (3, 3)
        x = conv2d(
            x,
            filters,
            config,
            kernel_size=kernel_size,
            activation=activation,
            conv_name='expand_conv2d_' + str(conv_index),
            bn_name='batch_normalization_' + str(bn_index),
            train=train)
        bn_index += 1
        conv_index += 1
      # Depthwise Convolution
      if use_depthwise:
        x = conv2d(x,
                   conv_filters=x.shape[-1],  # Depthwise conv
                   config=config,
                   kernel_size=block.kernel_size,
                   strides=block.strides,
                   activation=activation,
                   depthwise=True,
                   conv_name='depthwise_conv2d',
                   bn_name='batch_normalization_' + str(bn_index),
                   train=train)
        bn_index += 1

    # Squeeze and Excitation phase
    if use_se:
      assert block.se_ratio is not None
      assert 0 < block.se_ratio <= 1
      x = SqueezeExcite(x, filters, block, config, train=train)

    # Output phase
    x = conv2d(
        x,
        block.output_filters,
        config,
        activation=None,
        conv_name='project_conv2d_' + str(conv_index),
        bn_name='batch_normalization_' + str(bn_index),
        train=train)
    conv_index += 1

    if (block.id_skip and all(s == 1 for s in block.strides) and
        block.input_filters == block.output_filters):
      if drop_connect_rate and drop_connect_rate > 0:
        survival_probability = 1 - drop_connect_rate
        x = stochastic_depth(x, survival_probability, deterministic=not train)
      x = x + inputs

    return x


class Stem(flax.nn.Module):
  """Initial block of Efficientnet."""

  def apply(self,
            x: jnp.ndarray,
            config: ModelConfig,
            train: bool = True) -> jnp.ndarray:
    """Returns the output of the stem block.

    Args:
      x: The input to the block.
      config: ModelConfig, a set of model parameters.
      train: Whether we are training or predicting.
    """
    resolution = config.resolution
    if x.shape[1:3] != (resolution, resolution):
      raise ValueError('Wrong input size. Model was expecting ' +
                       'resolution {} '.format((resolution, resolution)) +
                       'but got input of resolution {}'.format(x.shape[1:3]))

    # Build stem
    x = conv2d(
        x,
        round_filters(config.stem_base_filters, config),
        config,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation=config.activation,
        train=train)
    return x


class Head(flax.nn.Module):
  """Final block of Efficientnet."""

  def apply(self,
            x: jnp.ndarray,
            config: ModelConfig,
            num_classes: int,
            train: bool = True) -> jnp.ndarray:
    """Returns the output of the head block.

    Args:
      x: The input to the block.
      config: A set of model parameters.
      num_classes: Dimension of the output of the model.
      train: Whether we are training or predicting.
    """
    # Build top
    x = conv2d(
        x,
        round_filters(config.top_base_filters, config),
        config,
        activation=config.activation,
        train=train)

    # Build classifier
    x = flax.nn.avg_pool(x, x.shape[1:3])
    if config.dropout_rate and config.dropout_rate > 0:
      x = flax.nn.dropout(x, config.dropout_rate, deterministic=not train)
    x = flax.nn.Dense(
        x, num_classes, kernel_init=dense_kernel_init_fn, name='dense')
    x = x.reshape([x.shape[0], -1])
    return x


class EfficientNet(flax.nn.Module):
  """Implements EfficientNet model."""

  def apply(self,
            x: jnp.ndarray,
            config: ModelConfig,
            num_classes: int = 1000,
            train: bool = True) -> jnp.ndarray:
    """Returns the output of the EfficientNet model.

    Args:
      x: The input batch of images.
      config: The model config.
      num_classes: Dimension of the output layer.
      train: Whether we are in training or inference.

    Returns:
      The output of efficientnet
    """
    config = copy.deepcopy(config)
    depth_coefficient = config.depth_coefficient
    blocks = config.blocks
    drop_connect_rate = config.drop_connect_rate

    resolution = config.resolution
    if x.shape[1:3] != (resolution, resolution):
      raise ValueError('Wrong input size. Model was expecting ' +
                       'resolution {} '.format((resolution, resolution)) +
                       'but got input of resolution {}'.format(x.shape[1:3]))

    # Build stem
    x = Stem(x, config, train=train)

    # Build blocks
    num_blocks_total = sum(
        round_repeats(block.num_repeat, depth_coefficient) for block in blocks)
    block_num = 0

    for block in blocks:
      assert block.num_repeat > 0
      # Update block input and output filters based on depth multiplier
      block.input_filters = round_filters(block.input_filters, config)
      block.output_filters = round_filters(block.output_filters, config)
      block.num_repeat = round_repeats(block.num_repeat, depth_coefficient)

      # The first block needs to take care of stride and filter size increase
      drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
      config.drop_connect_rate = drop_rate
      x = MBConvBlock(x, block, config, train=train)
      block_num += 1
      if block.num_repeat > 1:
        block.input_filters = block.output_filters
        block.strides = [1, 1]

        for _ in range(block.num_repeat - 1):
          drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
          config.drop_connect_rate = drop_rate
          x = MBConvBlock(x, block, config, train=train)
          block_num += 1

    # Build top
    x = Head(x, config, num_classes, train=train)

    return x
# pytype: enable=attribute-error


def get_efficientnet_module(model_name: str,
                            num_classes: int = 1000) -> EfficientNet:
  """Returns an EfficientNet module for a given architecture.

  Args:
    model_name: Name of the Efficientnet architecture to use (example:
      efficientnet-b0).
    num_classes: Dimension of the output layer.
  """
  return EfficientNet.partial(config=MODEL_CONFIGS[model_name],
                              num_classes=num_classes)
