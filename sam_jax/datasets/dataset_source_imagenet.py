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

"""Imagenet DatasetSource.

Initially forked from:
https://github.com/google/flax/blob/master/examples/imagenet/input_pipeline.py
"""

from typing import Dict, Tuple

from absl import flags
from absl import logging
import jax
from sam.autoaugment import autoaugment
from sam.sam_jax.datasets import dataset_source
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp


FLAGS = flags.FLAGS


flags.DEFINE_integer('randaug_num_layers', 2,
                     'Number of augmentations applied to each images by '
                     'RandAugment. Typical value is 2 and is generally not '
                     'changed.')
flags.DEFINE_integer('randaug_magnitude', 9,
                     'Magnitude of augmentations applied by RandAugment.')
flags.DEFINE_float('imagenet_mixup_alpha', 0.0, 'If > 0, use mixup.')


TRAIN_IMAGES = 1281167
EVAL_IMAGES = 50000

IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def _distorted_bounding_box_crop(image_bytes: tf.Tensor,
                                 bbox: tf.Tensor,
                                 min_object_covered: float = 0.1,
                                 aspect_ratio_range: Tuple[float,
                                                           float] = (0.75,
                                                                     1.33),
                                 area_range: Tuple[float, float] = (0.05, 1.0),
                                 max_attempts: int = 100) -> tf.Tensor:
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
  Returns:
    cropped image `Tensor`
  """
  shape = tf.image.extract_jpeg_shape(image_bytes)
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

  return image


def _resize(image: tf.Tensor, image_size: int) -> tf.Tensor:
  """Returns the resized image."""
  return tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[0]


def _at_least_x_are_equal(a: tf.Tensor, b: tf.Tensor,
                          x: int) -> tf.Tensor:
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes: tf.Tensor,
                            image_size: int) -> tf.Tensor:
  """Make a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = _distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10)
  original_shape = tf.image.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(bad, lambda: _decode_and_center_crop(image_bytes, image_size),
                  lambda: _resize(image, image_size))

  return image


def _decode_and_center_crop(image_bytes: tf.Tensor,
                            image_size: int) -> tf.Tensor:
  """Crops to center of image with padding then scales image_size."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = _resize(image, image_size)

  return image


def normalize_image(image: tf.Tensor) -> tf.Tensor:
  """Returns the normalized image.

  Image is normalized so that the mean and variance of each channel over the
  dataset is 0 and 1.

  Args:
    image: An image from the Imagenet dataset to normalize.
  """
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image


def preprocess_for_train(image_bytes: tf.Tensor,
                         dtype: tf.DType = tf.float32,
                         image_size: int = IMAGE_SIZE,
                         use_autoaugment: bool = False) -> tf.Tensor:
  """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: Data type of the returned image.
    image_size: Size of the returned image.
    use_autoaugment: If True, will apply autoaugment to the inputs.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_random_crop(image_bytes, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  if use_autoaugment:
    logging.info('Using autoaugment.')
    image = tf.cast(image, tf.uint8)
    image = autoaugment.distort_image_with_randaugment(image,
                                                       FLAGS.randaug_num_layers,
                                                       FLAGS.randaug_magnitude)
    image = tf.cast(image, tf.float32)
  image = tf.image.random_flip_left_right(image)
  image = normalize_image(image)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


def preprocess_for_eval(image_bytes: tf.Tensor,
                        dtype: tf.DType = tf.float32,
                        image_size: int = IMAGE_SIZE) -> tf.Tensor:
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: Data type of the returned image.
    image_size: Size of the returned image.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = normalize_image(image)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


def load_split(train: bool,
               cache: bool) -> tf.data.Dataset:
  """Creates a split from the ImageNet dataset using TensorFlow Datasets.

  Args:
    train: Whether to load the train or evaluation split.
    cache: Whether to cache the dataset.
  Returns:
    A `tf.data.Dataset`.
  """
  if train:
    split_size = TRAIN_IMAGES // jax.host_count()
    start = jax.host_id() * split_size
    split = 'train[{}:{}]'.format(start, start + split_size)
  else:
    # For validation, we load up the dataset on each host. This will have the
    # effect of evaluating on the whole dataset num_host times, but will
    # prevent size issues. This makes the performance slightly worse when
    # evaluating often, but spares us the need to pad the datasets and mask the
    # loss accordingly.
    split = 'validation'

  ds = tfds.load('imagenet2012:5.*.*', split=split, decoders={
      'image': tfds.decode.SkipDecoding(),
  })
  ds.options().experimental_threading.private_threadpool_size = 48
  ds.options().experimental_threading.max_intra_op_parallelism = 1

  if cache:
    ds = ds.cache()

  return ds


def mixup(batch: Dict[str, tf.Tensor], alpha: float) -> Dict[str, tf.Tensor]:
  """Generates augmented images using Mixup.

  Arguments:
    batch: Feature dict containing the images and the labels.
    alpha: Float that controls the strength of Mixup regularization.

  Returns:
    A feature dict containing the mix-uped images.
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


class Imagenet(dataset_source.DatasetSource):
  """Class that loads, preprocess and serves the Imagenet dataset."""

  def __init__(self, batch_size: int, image_size: int,
               image_level_augmentations: str = 'none'):
    """Instantiates the Imagenet dataset source.

    Args:
      batch_size: Global batch size used to train the model.
      image_size: Size to which the images should be resized (in number of
        pixels).
      image_level_augmentations: If set to 'autoaugment', will apply
        RandAugment to the training set.
    """
    self.batch_size = batch_size
    self.image_size = image_size
    self.num_training_obs = TRAIN_IMAGES
    self._train_ds = load_split(train=True, cache=True)
    self._test_ds = load_split(train=False, cache=True)
    self._num_classes = 1000
    self._image_level_augmentations = image_level_augmentations

  def get_train(self, use_augmentations: bool) -> tf.data.Dataset:
    """Returns the training set.

    The training set will be batched, and the remainder of the batch will be
    dropped (except if use_augmentation is False, in which case we don't drop
    the remainder as we are most likely computing the accuracy on the train
    set).

    Args:
      use_augmentations: Whether we should apply data augmentation (and possibly
        cutout) or not.
    """
    ds = self._train_ds.shuffle(16 * self.batch_size)
    ds = ds.map(lambda d: self.decode_example(  # pylint:disable=g-long-lambda
        d, use_augmentations=use_augmentations),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    batched = ds.batch(self.batch_size, drop_remainder=use_augmentations)
    if use_augmentations and FLAGS.imagenet_mixup_alpha > 0.0:
      batched = batched.map(lambda b: mixup(b, FLAGS.imagenet_mixup_alpha),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return batched

  def get_test(self) -> tf.data.Dataset:
    """Returns test set."""
    ds = self._test_ds.map(
        lambda d: self.decode_example(  # pylint:disable=g-long-lambda
            d, use_augmentations=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds.batch(self.batch_size, drop_remainder=False)

  def decode_example(self, example: Dict[str, tf.Tensor],
                     use_augmentations: bool) -> Dict[str, tf.Tensor]:
    """Decodes the raw examples from the imagenet tensorflow dataset.

    Args:
      example: A feature dict as returned by the tensorflow imagenet dataset.
      use_augmentations: Whether to use train time data augmentation or not.

    Returns:
      A dictionnary with an 'image' tensor and a one hot encoded 'label' tensor.
    """
    if use_augmentations:
      image = preprocess_for_train(
          example['image'],
          image_size=self.image_size,
          use_autoaugment=self._image_level_augmentations == 'autoaugment')
    else:
      image = preprocess_for_eval(example['image'], image_size=self.image_size)
    label = tf.one_hot(
        example['label'], depth=self._num_classes, on_value=1.0, off_value=0.0)
    return {'image': image, 'label': label}
