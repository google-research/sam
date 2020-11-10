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

"""Tests for sam.sam_jax.efficientnet.efficientnet."""

from absl.testing import absltest
import jax
import numpy as np
from sam.sam_jax.efficientnet import efficientnet
from sam.sam_jax.imagenet_models import load_model


class EfficientnetTest(absltest.TestCase):

  def _test_model_params(
      self, model_name: str, image_size: int, expected_params: int):
    module = efficientnet.get_efficientnet_module(model_name)
    model, _ = load_model.create_image_model(jax.random.PRNGKey(0),
                                             batch_size=1,
                                             image_size=image_size,
                                             module=module)
    num_params = sum(np.prod(e.shape) for e in jax.tree_leaves(model))
    self.assertEqual(num_params, expected_params)

  def test_efficientnet_b0(self):
    self._test_model_params('efficientnet-b0', 224, expected_params=5288548)

  def test_efficientnet_b1(self):
    self._test_model_params('efficientnet-b1', 240, expected_params=7794184)

  def test_efficientnet_b2(self):
    self._test_model_params('efficientnet-b2', 260, expected_params=9109994)

  def test_efficientnet_b3(self):
    self._test_model_params('efficientnet-b3', 300, expected_params=12233232)

  def test_efficientnet_b4(self):
    self._test_model_params('efficientnet-b4', 380, expected_params=19341616)

  def test_efficientnet_b5(self):
    self._test_model_params('efficientnet-b5', 456, expected_params=30389784)

  def test_efficientnet_b6(self):
    self._test_model_params('efficientnet-b6', 528, expected_params=43040704)

  def test_efficientnet_b7(self):
    self._test_model_params('efficientnet-b7', 600, expected_params=66347960)


if __name__ == '__main__':
  absltest.main()
