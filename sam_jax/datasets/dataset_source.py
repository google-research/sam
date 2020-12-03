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

"""Utility class to load datasets and apply data augmentation."""

import abc
from typing import Callable, Dict, Optional

from absl import flags
from absl import logging
import jax
from sam.sam_jax.datasets import augmentation
import tensorflow as tf
import tensorflow_datasets as tfds


FLAGS = flags.FLAGS


flags.DEFINE_bool('use_test_set', True,
                  'Whether to use the test set or not. If not, then 10% '
                  'observations will be set aside from the training set and '
                  'used as a validation set instead.')


class DatasetSource(abc.ABC):
  """Parent for classes that load, preprocess and serve datasets.

  Child class constructor should set a `num_training_obs` and a `batch_size`
  attribute.
  """
  batch_size = ...  # type: int
  num_training_obs = ...  # type: int

  @abc.abstractmethod
  def get_train(self, use_augmentations: bool) -> tf.data.Dataset:
    """Returns the training set.

    The training set will be batched, and the remainder of the batch will be
    dropped (except if use_augmentation is False, in which case we don't drop
    the remainder as we are most likely computing the accuracy on the train set.

    Args:
      use_augmentations: Whether we should apply data augmentation (and possibly
        cutout) or not.
    """

  @abc.abstractmethod
  def get_test(self) -> tf.data.Dataset:
    """Returns test set."""


def _resize(image: tf.Tensor, image_size: int, method: Optional[str] = None):
  if method is not None:
    return tf.image.resize(image, [image_size, image_size], method)
  return tf.compat.v1.image.resize_bicubic(image, [image_size, image_size])


class TFDSDatasetSource(DatasetSource):
  """Parent for classes that load, preprocess and serve TensorFlow datasets.

  Small datasets like CIFAR, SVHN and Fashion MNIST subclass TFDSDatasetSource.
  """
  batch_size = ...  # type: int
  num_training_obs = ...  # type: int
  _train_ds = ...  # type: tf.data.Dataset
  _test_ds = ...  # type: tf.data.Dataset
  _augmentation = ...  # type: str
  _num_classes = ...  # type: int
  _image_mean = ...  # type: tf.tensor
  _image_std = ...  # type: tf.tensor
  _dataset_name = ...  # type: str
  _batch_level_augmentations = ...  # type: Callable
  _image_size = ...  # type: Optional[int]

  def _apply_image_augmentations(
      self, example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    if self._augmentation in ['autoaugment', 'aa-only']:
      example = augmentation.auto_augmentation(example, self._dataset_name)
    if self._augmentation in ['basic', 'autoaugment']:
      example = augmentation.weak_image_augmentation(example)
    return example

  def _preprocess_batch(self,
                        examples: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    image, label = examples['image'], examples['label']
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - self._image_mean) / self._image_std
    label = tf.one_hot(
        label, depth=self._num_classes, on_value=1.0, off_value=0.0)
    return {'image': image, 'label': label}

  def get_train(self, use_augmentations: bool) -> tf.data.Dataset:
    """Returns the training set.

    The training set will be batched, and the remainder of the batch will be
    dropped (except if use_augmentations is False, in which case we don't drop
    the remainder as we are most likely computing the accuracy on the train
    set).

    Args:
      use_augmentations: Whether we should apply data augmentation (and possibly
        cutout) or not.
    """
    ds = self._train_ds.shuffle(50000)
    if use_augmentations:
      ds = ds.map(self._apply_image_augmentations,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Don't drop remainder if we don't use augmentation, as we are evaluating.
    ds = ds.batch(self.batch_size, drop_remainder=use_augmentations)
    ds = ds.map(self._preprocess_batch,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if self._batch_level_augmentations and use_augmentations:
      ds = ds.map(self._batch_level_augmentations,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if self._image_size:
      def resize(batch):
        image = _resize(batch['image'], self._image_size)
        return {'image': image, 'label': batch['label']}
      ds = ds.map(resize)
    return ds

  def get_test(self) -> tf.data.Dataset:
    """Returns the batched test set."""
    eval_batch_size = min(32, self.batch_size)
    ds = self._test_ds.batch(eval_batch_size).map(
        self._preprocess_batch,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if self._image_size:
      def resize(batch):
        image = _resize(batch['image'], self._image_size)
        return {'image': image, 'label': batch['label']}
      ds = ds.map(resize)
    return ds


class CifarDatasetSource(TFDSDatasetSource):
  """Parent class for DatasetSource created from cifar10/cifar100 datasets.

  The child class constructor must set _num_classes (integer, number of classes
  in the dataset).
  """

  def __init__(self, batch_size: int, name: str, image_level_augmentations: str,
               batch_level_augmentations: str,
               image_size: Optional[int] = None):
    """Instantiates the DatasetSource.

    Args:
      batch_size: Batch size to use for training and evaluation.
      name: Name of the Tensorflow Dataset to use. Should be cifar10 or
        cifar100.
      image_level_augmentations: Augmentations to apply to the images. Should be
        one of:
        * none: No augmentations are applied.
        * basic: Applies random crops and horizontal translations.
        * autoaugment: Applies the best found policy for Cifar from the
          AutoAugment paper.
      batch_level_augmentations: Augmentations to apply at the batch level. Only
        cutout is needed to get SOTA results. The following are implemented:
        * none: No augmentations are applied.
        * cutout: Applies cutout (https://arxiv.org/abs/1708.04552).
        * mixup: Applies mixup (https://arxiv.org/pdf/1710.09412.pdf).
        * mixcut: Applies mixup and cutout.
      image_size: Size to which the image should be rescaled. If None, the
        standard size is used (32x32).
    """
    assert name in ['cifar10', 'cifar100']
    assert image_level_augmentations in ['none', 'basic', 'autoaugment']
    assert batch_level_augmentations in ['none', 'cutout']
    self._image_size = image_size
    self.batch_size = batch_size
    if FLAGS.use_test_set:
      self.num_training_obs = 50000
      train_split_size = self.num_training_obs // jax.host_count()
      start = jax.host_id() * train_split_size
      train_split = 'train[{}:{}]'.format(start, start + train_split_size)
      self._train_ds = tfds.load(name, split=train_split).cache()
      self._test_ds = tfds.load(name, split='test').cache()
      logging.info('Used test set instead of validation set.')
    else:
      # Validation split not implemented for multi-host training.
      assert jax.host_count() == 1
      self._train_ds = tfds.load(name, split='train[:45000]').cache()
      self._test_ds = tfds.load(name, split='train[45000:]').cache()
      self.num_training_obs = 45000
      logging.info('Used validation set instead of test set.')
    self._augmentation = image_level_augmentations
    if batch_level_augmentations == 'cutout':
      self._batch_level_augmentations = augmentation.cutout
    elif batch_level_augmentations == 'mixup':
      self._batch_level_augmentations = augmentation.mixup
    elif batch_level_augmentations == 'mixcut':
      self._batch_level_augmentations = (
          lambda x: augmentation.cutout(augmentation.mixup(x)))
    else:
      self._batch_level_augmentations = None
    if name == 'cifar10':
      self._image_mean = tf.constant([[[0.49139968, 0.48215841, 0.44653091]]])
      self._image_std = tf.constant([[[0.24703223, 0.24348513, 0.26158784]]])
    else:
      self._image_mean = tf.constant([[[0.50707516, 0.48654887, 0.44091784]]])
      self._image_std = tf.constant([[[0.26733429, 0.25643846, 0.27615047]]])
    self._num_classes = None  # To define in child classes


class Cifar10(CifarDatasetSource):
  """Cifar10 DatasetSource."""

  def __init__(self, batch_size: int, image_level_augmentations: str,
               batch_level_augmentations: str, image_size: int = None):
    """See parent class for more information."""
    super().__init__(batch_size, 'cifar10', image_level_augmentations,
                     batch_level_augmentations, image_size)
    self._num_classes = 10
    self._dataset_name = 'cifar10'


class Cifar100(CifarDatasetSource):
  """Cifar100 DatasetSource."""

  def __init__(self, batch_size: int, image_level_augmentations: str,
               batch_level_augmentations: str, image_size: int = None):
    """See parent class for more information."""
    super().__init__(batch_size, 'cifar100', image_level_augmentations,
                     batch_level_augmentations, image_size)
    self._num_classes = 100
    self._dataset_name = 'cifar100'


class FashionMnist(TFDSDatasetSource):
  """Fashion Mnist dataset."""

  def __init__(self, batch_size: int, image_level_augmentations: str,
               batch_level_augmentations: str):
    """Instantiates the DatasetSource.

    Args:
      batch_size: Batch size to use for training and evaluation.
      image_level_augmentations: Augmentations to apply to the images. Should be
        one of:
        * none: No augmentations are applied.
        * basic: Applies random crops and horizontal translations.
      batch_level_augmentations: Augmentations to apply at the batch level.
        * none: No augmentations are applied.
        * cutout: Applies cutout (https://arxiv.org/abs/1708.04552).
    """
    assert image_level_augmentations in ['none', 'basic']
    assert batch_level_augmentations in ['none', 'cutout']
    self.batch_size = batch_size
    self._image_size = None
    if FLAGS.use_test_set:
      self._train_ds = tfds.load('fashion_mnist', split='train').cache()
      self._test_ds = tfds.load('fashion_mnist', split='test').cache()
      logging.info('Used test set instead of validation set.')
      self.num_training_obs = 60000
    else:
      self._train_ds = tfds.load('fashion_mnist', split='train[:54000]').cache()
      self._test_ds = tfds.load('fashion_mnist', split='train[54000:]').cache()
      self.num_training_obs = 54000
      logging.info('Used validation set instead of test set.')
    self._augmentation = image_level_augmentations
    if batch_level_augmentations == 'cutout':
      self._batch_level_augmentations = augmentation.cutout
    else:
      self._batch_level_augmentations = None
    self._image_mean = tf.constant([[[0.1307]]])
    self._image_std = tf.constant([[[0.3081]]])
    self._num_classes = 10
    self._dataset_name = 'fashion_mnist'


class SVHN(TFDSDatasetSource):
  """SVHN dataset."""

  def __init__(self, batch_size: int, image_level_augmentations: str,
               batch_level_augmentations: str):
    """Instantiates the DatasetSource.

    Args:
      batch_size: Batch size to use for training and evaluation.
      image_level_augmentations: Augmentations to apply to the images. Should be
        one of:
        * none: No augmentations are applied.
        * basic: Applies random crops and horizontal translations.
        * autoaugment: Applies the best found policy for SVHN from the
          AutoAugment paper. Also applies the basic augmentations on top of it.
        * aa-only: Same as autoaugment but doesn't apply the basic
          augmentations. Should be preferred for SVHN.
      batch_level_augmentations: Augmentations to apply at the batch level.
        * none: No augmentations are applied.
        * cutout: Applies cutout (https://arxiv.org/abs/1708.04552).
    """
    assert image_level_augmentations in [
        'none', 'basic', 'autoaugment', 'aa-only']
    assert batch_level_augmentations in ['none', 'cutout']
    self.batch_size = batch_size
    self._image_size = None
    if FLAGS.use_test_set:
      ds_base = tfds.load('svhn_cropped', split='train')
      ds_extra = tfds.load('svhn_cropped', split='extra')
      self._train_ds = ds_base.concatenate(ds_extra).cache()
      self._test_ds = tfds.load('svhn_cropped', split='test').cache()
      logging.info('Used test set instead of validation set.')
      self.num_training_obs = 73257+531131
    else:
      ds_base = tfds.load('svhn_cropped', split='train[:65929]')
      ds_extra = tfds.load('svhn_cropped', split='extra')
      self._train_ds = ds_base.concatenate(ds_extra).cache()
      self._test_ds = tfds.load('svhn_cropped', split='train[65929:]').cache()
      self.num_training_obs = 65929+531131
      logging.info('Used validation set instead of test set.')
    self._augmentation = image_level_augmentations
    if batch_level_augmentations == 'cutout':
      self._batch_level_augmentations = augmentation.cutout
    else:
      self._batch_level_augmentations = None
    self._image_mean = tf.constant([[[0.43090966, 0.4302428, 0.44634357]]])
    self._image_std = tf.constant([[[0.19759192, 0.20029082, 0.19811132]]])
    self._num_classes = 10
    self._dataset_name = 'svhn'
