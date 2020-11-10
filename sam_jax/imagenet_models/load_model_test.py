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

"""Tests for sam.sam_jax.imagenet_models."""

from absl.testing import absltest
from absl.testing import parameterized
import flax
import numpy as np
from sam.sam_jax.imagenet_models import load_model


class LoadModelTest(parameterized.TestCase):

  def test_CreateEfficientnetModel(self):
    model, state = load_model.get_model('efficientnet-b0', 1, 224, 1000)
    self.assertIsInstance(model, flax.nn.Model)
    self.assertIsInstance(state, flax.nn.Collection)
    fake_input = np.zeros([1, 224, 224, 3])
    with flax.nn.stateful(state, mutable=False):
      logits = model(fake_input, train=False)
    self.assertEqual(logits.shape, (1, 1000))

  def test_CreateResnetModel(self):
    model, state = load_model.get_model('Resnet50', 1, 224, 1000)
    self.assertIsInstance(model, flax.nn.Model)
    self.assertIsInstance(state, flax.nn.Collection)
    fake_input = np.zeros([1, 224, 224, 3])
    with flax.nn.stateful(state, mutable=False):
      logits = model(fake_input, train=False)
    self.assertEqual(logits.shape, (1, 1000))


if __name__ == '__main__':
  absltest.main()
