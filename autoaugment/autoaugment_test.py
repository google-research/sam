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

"""Tests for sam.autoaugment.autoaugment."""

from absl.testing import absltest
from absl.testing import parameterized
from sam.autoaugment import autoaugment
import tensorflow as tf


class AutoAugmentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('ShearX', 'ShearX'),
      ('ShearY', 'ShearY'),
      ('Cutout', 'Cutout'),
      ('TranslateX', 'TranslateX'),
      ('TranslateY', 'TranslateY'),
      ('Rotate', 'Rotate'),
      ('AutoContrast', 'AutoContrast'),
      ('Invert', 'Invert'),
      ('Equalize', 'Equalize'),
      ('Solarize', 'Solarize'),
      ('Posterize', 'Posterize'),
      ('Contrast', 'Contrast'),
      ('Color', 'Color'),
      ('Brightness', 'Brightness'),
      ('Sharpness', 'Sharpness'))
  def test_image_processing_function(self, name: str):
    hparams = autoaugment.HParams(cutout_const=10, translate_const=25)
    replace_value = [128, 128, 128]
    function, _, args = autoaugment._parse_policy_info(
        name, 1.0, 10, replace_value, hparams)
    cifar_image_shape = [32, 32, 3]
    image = tf.zeros(cifar_image_shape, tf.uint8)
    augmented_image = function(image, *args)
    self.assertEqual(augmented_image.shape, cifar_image_shape)
    self.assertEqual(augmented_image.dtype, tf.uint8)

  @parameterized.named_parameters(('cifar', 'cifar'), ('svhn', 'svhn'),
                                  ('imagenet', 'imagenet'))
  def test_autoaugment_function(self, dataset_name):
    autoaugment_fn = lambda image: autoaugment.distort_image_with_autoaugment(  # pylint:disable=g-long-lambda
        image, dataset_name)
    image_shape = [224, 224, 3] if dataset_name == 'imagenet' else [32, 32, 3]
    image = tf.zeros(image_shape, tf.uint8)
    augmented_image = autoaugment_fn(image)
    self.assertEqual(augmented_image.shape, image_shape)
    self.assertEqual(augmented_image.dtype, tf.uint8)

if __name__ == '__main__':
  absltest.main()
