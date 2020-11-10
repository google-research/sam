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

"""Implements data augmentations for cifar10/cifar100."""

from typing import Dict

from absl import flags
from sam.autoaugment import autoaugment
import tensorflow as tf
import tensorflow_probability as tfp


FLAGS = flags.FLAGS


flags.DEFINE_integer('cutout_length', 16,
                     'Length (in pixels) of the cutout patch. Default value of '
                     '16 is used to get SOTA on cifar10/cifar100')


def weak_image_augmentation(example: Dict[str, tf.Tensor],
                            random_crop_pad: int = 4) -> Dict[str, tf.Tensor]:
  """Applies random crops and horizontal flips.

  Simple data augmentations that are (almost) always used with cifar. Pad the
  image with `random_crop_pad` before randomly cropping it to its original
  size. Also randomly apply horizontal flip.

  Args:
    example: An example dict containing an image and a label.
    random_crop_pad: By how many pixels should the image be padded on each side
      before cropping.

  Returns:
    An example with the same label and an augmented version of the image.
  """
  image, label = example['image'], example['label']
  image = tf.image.random_flip_left_right(image)
  image_shape = tf.shape(image)
  image = tf.pad(
      image, [[random_crop_pad, random_crop_pad],
              [random_crop_pad, random_crop_pad], [0, 0]],
      mode='REFLECT')
  image = tf.image.random_crop(image, image_shape)
  return {'image': image, 'label': label}


def auto_augmentation(example: Dict[str, tf.Tensor],
                      dataset_name: str) -> Dict[str, tf.Tensor]:
  """Applies the AutoAugment policy found for the dataset.

  AutoAugment: Learning Augmentation Policies from Data
  https://arxiv.org/abs/1805.09501

  Args:
    example: An example dict containing an image and a label.
    dataset_name: Name of the dataset for which we should return the optimal
      policy. Should be 'cifar[10|100]', 'svhn' or 'imagenet'.

  Returns:
    An example with the same label and an augmented version of the image.
  """
  if dataset_name in ('cifar10', 'cifar100'): dataset_name = 'cifar'
  image, label = example['image'], example['label']
  image = autoaugment.distort_image_with_autoaugment(image, dataset_name)
  return {'image': image, 'label': label}


def cutout(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
  """Applies cutout to a batch of images.

  The cut out patch will be replaced by zeros (thus the batch should be
  normalized before cutout is applied).

  Reference:
  Improved Regularization of Convolutional Neural Networks with Cutout
  https://arxiv.org/abs/1708.04552

  Implementation inspired by:
  third_party/cloud_tpu/models/efficientnet/autoaugment.py

  Args:
    batch: A batch of images and labels.

  Returns:
    The same batch where cutout has been applied to the images.
  """
  length, replace = FLAGS.cutout_length, 0.0
  images, labels = batch['image'], batch['label']
  num_channels = tf.shape(images)[3]
  image_height, image_width = tf.shape(images)[1], tf.shape(images)[2]

  cutout_center_height = tf.random.uniform(
      shape=[], minval=0, maxval=image_height,
      dtype=tf.int32)
  cutout_center_width = tf.random.uniform(
      shape=[], minval=0, maxval=image_width,
      dtype=tf.int32)

  lower_pad = tf.maximum(0, cutout_center_height - length // 2)
  upper_pad = tf.maximum(0, image_height - cutout_center_height - length // 2)
  left_pad = tf.maximum(0, cutout_center_width - length // 2)
  right_pad = tf.maximum(0, image_width - cutout_center_width - length // 2)

  cutout_shape = [image_height - (lower_pad + upper_pad),
                  image_width - (left_pad + right_pad)]

  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]

  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=images.dtype),
      padding_dims, constant_values=1)

  patch = tf.ones_like(images, dtype=images.dtype) * replace,

  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, num_channels])

  images = tf.where(
      tf.equal(mask, 0),
      patch,
      images)

  images = tf.squeeze(images, axis=0)

  return {'image': images, 'label': labels}


def mixup(batch: Dict[str, tf.Tensor],
          alpha: float = 1.0) -> Dict[str, tf.Tensor]:
  """Generates augmented images using Mixup.

  Arguments:
    batch: Feature dict containing the images and the labels.
    alpha: Float that controls the strength of Mixup regularization.

  Returns:
    A feature dict containing the images and labels augmented with mixup.
  """
  images, labels = batch['image'], batch['label']
  batch_size = 1  # Unique mixing parameter for all samples
  mix_weight = tfp.distributions.Beta(alpha, alpha).sample([batch_size, 1])
  mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
  images_mix_weight = tf.reshape(mix_weight, [batch_size, 1, 1, 1])
  images_mix = (
      images * images_mix_weight + images[::-1] * (1. - images_mix_weight))
  labels_mix = labels * mix_weight + labels[::-1] * (1. - mix_weight)
  return {'image': images_mix, 'label': labels_mix}
