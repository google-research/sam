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

"""Tests for google3.learning.neurosurgeon.research.image_classification.efficientnet.optim."""

from absl.testing import absltest
from jax.config import config
import numpy as onp
from sam.sam_jax.efficientnet import optim
import tensorflow.compat.v1 as tf


# Use double precision for better comparison with Tensorflow version.
config.update("jax_enable_x64", True)


class OptimTest(tf.test.TestCase):

  def test_RMSProp(self):
    """Updates should match Tensorflow1 behavior."""
    lr, mom, rho = 0.5, 0.7, 0.8
    onp.random.seed(0)
    w0 = onp.random.normal(size=[17, 13, 1])
    num_steps = 10
    # First compute weights updates for TF1 version.
    tf1_updated_weights = []
    with tf.Session() as sess:
      var0 = tf.Variable(w0, trainable=True)
      opt = tf.train.RMSPropOptimizer(
          learning_rate=lr, decay=rho, momentum=mom, epsilon=0.001)
      loss = lambda: (var0**2) / 2.0
      step = opt.minimize(loss, var_list=[var0])
      sess.run(tf.global_variables_initializer())
      for _ in range(num_steps):
        sess.run(step)
        tf1_updated_weights.append(sess.run(var0))
    # Now compute the updates for FLAX version.
    flax_updated_weights = []
    optimizer_def = optim.RMSProp(
        learning_rate=lr, beta=mom, beta2=rho, eps=0.001)
    ref_opt = optimizer_def.create(w0)
    for _ in range(num_steps):
      gradient = ref_opt.target
      ref_opt = ref_opt.apply_gradient(gradient)
      flax_updated_weights.append(ref_opt.target)
    for a, b in zip(tf1_updated_weights, flax_updated_weights):
      self.assertAllClose(a, b)

  def test_RMSPropWithEMA(self):
    """Updates should match Tensorflow1 behavior."""
    lr, mom, rho, ema_decay = 0.05, 0.4, 0.8, 1.0
    onp.random.seed(0)
    w0 = onp.array([1.0])
    num_steps = 10
    # First compute weights updates for TF1 version.
    tf1_updated_weights = []
    with tf.Session() as sess:
      global_step = tf.train.get_or_create_global_step()
      ema = tf.train.ExponentialMovingAverage(
          decay=ema_decay, num_updates=global_step)
      var0 = tf.Variable(w0, trainable=True)
      opt = tf.train.RMSPropOptimizer(
          learning_rate=lr, decay=rho, momentum=mom, epsilon=0.000)
      loss = lambda: (var0**2) / 2.0
      step = opt.minimize(loss, var_list=[var0], global_step=global_step)
      with tf.control_dependencies([step]):
        step = ema.apply([var0])
      sess.run(tf.global_variables_initializer())
      for _ in range(num_steps):
        sess.run(step)
        tf1_updated_weights.append(sess.run(ema.average(var0)))
    # Now computes the updates for FLAX version.
    flax_updated_weights = []
    optimizer_def = optim.RMSProp(
        learning_rate=lr, beta=mom, beta2=rho, eps=0.000)
    ref_opt = optimizer_def.create(w0)
    ema = optim.ExponentialMovingAverage(w0, ema_decay, 0)
    for _ in range(num_steps):
      gradient = ref_opt.target
      ref_opt = ref_opt.apply_gradient(gradient)
      ema = ema.update_moving_average(ref_opt.target, ref_opt.state.step)
      flax_updated_weights.append(ema.param_ema)
    for a, b in zip(tf1_updated_weights, flax_updated_weights):
      self.assertAllClose(a, b)


if __name__ == "__main__":
  absltest.main()
