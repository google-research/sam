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

"""Tests for sam.sam_jax.imagenet_models.resnet.

Forked from
https://github.com/google/flax/blob/master/examples/imagenet/resnet_v1_test.py
"""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from sam.sam_jax.imagenet_models import resnet


class ResNetTest(absltest.TestCase):
  """Test cases for ResNet V1 model."""

  def test_resnet_v1_module(self):
    """Tests ResNet V1 model definition."""
    rng = jax.random.PRNGKey(0)
    model_def = resnet.ResNet50.partial(num_classes=10, dtype=jnp.float32)
    output, init_params = model_def.init_by_shape(
        rng, [((8, 224, 224, 3), jnp.float32)])

    self.assertEqual((8, 10), output.shape)
    self.assertLen(init_params, 19)


if __name__ == '__main__':
  absltest.main()
