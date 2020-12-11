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

"""Build FLAX models for image classification."""

from typing import Optional, Tuple

from absl import flags
import flax
from flax.training import checkpoints
import jax
from jax import numpy as jnp
from jax import random

from sam.sam_jax.efficientnet import efficientnet
from sam.sam_jax.imagenet_models import resnet


FLAGS = flags.FLAGS

flags.DEFINE_bool('from_pretrained_checkpoint', False,
                  'If True, the model will be restarted from an pretrained '
                  'checkpoint')
flags.DEFINE_string('efficientnet_checkpoint_path', None,
                    'If finetuning, path to the efficientnet checkpoint.')


_AVAILABLE_MODEL_NAMES = [
    'Resnet'
] + list(efficientnet.MODEL_CONFIGS.keys())


def create_image_model(
    prng_key: jnp.ndarray, batch_size: int, image_size: int,
    module: flax.nn.Module) -> Tuple[flax.nn.Model, flax.nn.Collection]:
  """Instantiates a FLAX model and its state.

  Args:
    prng_key: PRNG key to use to sample the initial weights.
    batch_size: Batch size that the model should expect.
    image_size: Dimension of the image (assumed to be squared).
    module: FLAX module describing the model to instantiates.

  Returns:
    A FLAX model and its state.
  """
  input_shape = (batch_size, image_size, image_size, 3)
  with flax.nn.stateful() as init_state:
    with flax.nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = module.init_by_shape(
          prng_key, [(input_shape, jnp.float32)])
      model = flax.nn.Model(module, initial_params)
  return model, init_state


class ModelNameError(Exception):
  """Exception to raise when the model name is not recognized."""
  pass


def _replace_dense_layer(model: flax.nn.Model, head: flax.nn.Model):
  """Replaces the last layer (head) of a model with the head of another one.

  Args:
    model: Model for which we should keep all layers except the head.
    head: Model from which we should copy the head.

  Returns:
    A model composed from the last layer of `head` and all the other layers of
      `model`.
  """
  new_params = {}
  for (ak, av), (bk, bv) in zip(
      flax.traverse_util.flatten_dict(model.params).items(),
      flax.traverse_util.flatten_dict(head.params).items()):
    if ak[1] == 'dense':
      new_params[bk] = bv
    else:
      new_params[ak] = av
  return head.replace(params=flax.traverse_util.unflatten_dict(new_params))


def get_model(
    model_name: str,
    batch_size: int,
    image_size: int,
    num_classes: int = 1000,
    prng_key: Optional[jnp.ndarray] = None
    ) -> Tuple[flax.nn.Model, flax.nn.Collection]:
  """Returns an initialized model of the chosen architecture.

  Args:
    model_name: Name of the architecture to use. See image_classification.train
      flags for a list of available models.
    batch_size: The batch size that the model should expect.
    image_size: Dimension of the image (assumed to be squared).
    num_classes: Dimension of the output layer. Should be 1000, but is left as
      an argument for consistency with other load_model functions. An error will
      be raised if num_classes is not 1000.
    prng_key: PRNG key to use to sample the weights.

  Returns:
    The initialized model and its state.

  Raises:
    ModelNameError: If the name of the architecture is not recognized.
  """
  if model_name == 'Resnet50':
    module = resnet.ResNet50.partial(num_classes=num_classes)
  elif model_name == 'Resnet101':
    module = resnet.ResNet101.partial(num_classes=num_classes)
  elif model_name == 'Resnet152':
    module = resnet.ResNet152.partial(num_classes=num_classes)
  elif model_name in efficientnet.MODEL_CONFIGS:
    module = efficientnet.get_efficientnet_module(
        model_name, num_classes=num_classes)
  else:
    raise ModelNameError('Unrecognized model name.')
  if not prng_key:
    prng_key = random.PRNGKey(0)

  model, init_state = create_image_model(prng_key, batch_size, image_size,
                                         module)

  if FLAGS.from_pretrained_checkpoint:
    if FLAGS.efficientnet_checkpoint_path is None:
      raise ValueError(
          'For finetuning, must set `efficientnet_checkpoint_path` to a '
          'valid efficientnet checkpoint.')
    # If the number of class is 1000, just load the imagenet/JFT checkpoint.
    if num_classes == 1000:
      model, init_state = checkpoints.restore_checkpoint(
          FLAGS.efficientnet_checkpoint_path,
          (model, init_state))
    # Else we need to change the size of the last layer (head):
    else:
      # Pretrained model on JFT/Imagenet.
      imagenet_module = efficientnet.get_efficientnet_module(
          model_name, num_classes=1000)
      imagenet_model, imagenet_state = create_image_model(
          prng_key, batch_size, image_size, imagenet_module)
      imagenet_model, imagenet_state = checkpoints.restore_checkpoint(
          FLAGS.efficientnet_checkpoint_path,
          (imagenet_model, imagenet_state))
      # Replace all the layers of the initialized model with the weights
      # extracted from the pretrained model.
      model = _replace_dense_layer(imagenet_model, model)
      init_state = imagenet_state

  return model, init_state
