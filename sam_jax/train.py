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

"""Trains a model on cifar10, cifar100, SVHN, F-MNIST or imagenet."""

import os

from absl import app
from absl import flags
from absl import logging
import jax
from sam.sam_jax.datasets import dataset_source as dataset_source_lib
from sam.sam_jax.datasets import dataset_source_imagenet
from sam.sam_jax.efficientnet import efficientnet
from sam.sam_jax.imagenet_models import load_model as load_imagenet_model
from sam.sam_jax.models import load_model
from sam.sam_jax.training_utils import flax_training
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile


FLAGS = flags.FLAGS

flags.DEFINE_enum('dataset', 'cifar10', [
    'cifar10', 'cifar100', 'fashion_mnist', 'svhn', 'imagenet', 'Birdsnap',
    'cifar100_brain', 'Stanford_Cars', 'Flowers', 'FGVC_Aircraft',
    'Oxford_IIIT_Pets', 'Food_101'
], 'Name of the dataset.')
flags.DEFINE_enum('model_name', 'WideResnet28x10', [
    'WideResnet28x10', 'WideResnet28x6_ShakeShake', 'Pyramid_ShakeDrop',
    'Resnet50', 'Resnet101', 'Resnet152'
] + list(efficientnet.MODEL_CONFIGS.keys()), 'Name of the model to train.')
flags.DEFINE_integer('num_epochs', 200,
                     'How many epochs the model should be trained for.')
flags.DEFINE_integer(
    'batch_size', 128, 'Global batch size. If multiple '
    'replicas are used, each replica will receive '
    'batch_size / num_replicas examples. Batch size should be divisible by '
    'the number of available devices.')
flags.DEFINE_string(
    'output_dir', '', 'Directory where the checkpoints and the tensorboard '
    'records should be saved.')
flags.DEFINE_enum(
    'image_level_augmentations', 'basic', ['none', 'basic', 'autoaugment',
                                           'aa-only'],
    'Augmentations applied to the images. Should be `none` for '
    'no augmentations, `basic` for the standard horizontal '
    'flips and random crops, and `autoaugment` for the best '
    'AutoAugment policy for cifar10. For SVHN, aa-only should be use for '
    'autoaugment without random crops or flips.'
    'For Imagenet, setting to autoaugment will use RandAugment. For '
    'FromBrainDatasetSource datasets, this flag is ignored.')
flags.DEFINE_enum(
    'batch_level_augmentations', 'none', ['none', 'cutout', 'mixup', 'mixcut'],
    'Augmentations that are applied at the batch level. '
    'Not used by Imagenet and FromBrainDatasetSource datasets.')


def main(_):

  tf.enable_v2_behavior()
  # make sure tf does not allocate gpu memory
  tf.config.experimental.set_visible_devices([], 'GPU')

  # Performance gains on TPU by switching to hardware bernoulli.
  def hardware_bernoulli(rng_key, p=jax.numpy.float32(0.5), shape=None):
    lax_key = jax.lax.tie_in(rng_key, 0.0)
    return jax.lax.rng_uniform(lax_key, 1.0, shape) < p

  def set_hardware_bernoulli():
    jax.random.bernoulli = hardware_bernoulli

  set_hardware_bernoulli()

  # As we gridsearch the weight decay and the learning rate, we add them to the
  # output directory path so that each model has its own directory to save the
  # results in. We also add the `run_seed` which is "gridsearched" on to
  # replicate an experiment several times.
  output_dir_suffix = os.path.join(
      'lr_' + str(FLAGS.learning_rate),
      'wd_' + str(FLAGS.weight_decay),
      'rho_' + str(FLAGS.sam_rho),
      'seed_' + str(FLAGS.run_seed))

  output_dir = os.path.join(FLAGS.output_dir, output_dir_suffix)

  if not gfile.exists(output_dir):
    gfile.makedirs(output_dir)

  num_devices = jax.local_device_count() * jax.host_count()
  assert FLAGS.batch_size % num_devices == 0
  local_batch_size = FLAGS.batch_size // num_devices
  info = 'Total batch size: {} ({} x {} replicas)'.format(
      FLAGS.batch_size, local_batch_size, num_devices)
  logging.info(info)

  if FLAGS.dataset == 'cifar10':
    if FLAGS.from_pretrained_checkpoint:
      image_size = efficientnet.name_to_image_size(FLAGS.model_name)
    else:
      image_size = None
    dataset_source = dataset_source_lib.Cifar10(
        FLAGS.batch_size // jax.host_count(),
        FLAGS.image_level_augmentations,
        FLAGS.batch_level_augmentations,
        image_size=image_size)
  elif FLAGS.dataset == 'cifar100':
    if FLAGS.from_pretrained_checkpoint:
      image_size = efficientnet.name_to_image_size(FLAGS.model_name)
    else:
      image_size = None
    dataset_source = dataset_source_lib.Cifar100(
        FLAGS.batch_size // jax.host_count(), FLAGS.image_level_augmentations,
        FLAGS.batch_level_augmentations, image_size=image_size)

  elif FLAGS.dataset == 'fashion_mnist':
    dataset_source = dataset_source_lib.FashionMnist(
        FLAGS.batch_size, FLAGS.image_level_augmentations,
        FLAGS.batch_level_augmentations)
  elif FLAGS.dataset == 'svhn':
    dataset_source = dataset_source_lib.SVHN(
        FLAGS.batch_size, FLAGS.image_level_augmentations,
        FLAGS.batch_level_augmentations)
  elif FLAGS.dataset == 'imagenet':
    imagenet_image_size = efficientnet.name_to_image_size(FLAGS.model_name)
    dataset_source = dataset_source_imagenet.Imagenet(
        FLAGS.batch_size // jax.host_count(), imagenet_image_size,
        FLAGS.image_level_augmentations)
  else:
    raise ValueError('Dataset not recognized.')

  if 'cifar' in FLAGS.dataset or 'svhn' in FLAGS.dataset:
    if image_size is None or 'svhn' in FLAGS.dataset:
      image_size = 32
    num_channels = 3
    num_classes = 100 if FLAGS.dataset == 'cifar100' else 10
  elif FLAGS.dataset == 'fashion_mnist':
    image_size = 28  # For Fashion Mnist
    num_channels = 1
    num_classes = 10
  elif FLAGS.dataset == 'imagenet':
    image_size = imagenet_image_size
    num_channels = 3
    num_classes = 1000
  else:
    raise ValueError('Dataset not recognized.')

  try:
    model, state = load_imagenet_model.get_model(FLAGS.model_name,
                                                 local_batch_size, image_size,
                                                 num_classes)
  except load_imagenet_model.ModelNameError:
    model, state = load_model.get_model(FLAGS.model_name,
                                        local_batch_size, image_size,
                                        num_classes, num_channels)

  # Learning rate will be overwritten by the lr schedule, we set it to zero.
  optimizer = flax_training.create_optimizer(model, 0.0)

  flax_training.train(optimizer, state, dataset_source, output_dir,
                      FLAGS.num_epochs)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  app.run(main)
