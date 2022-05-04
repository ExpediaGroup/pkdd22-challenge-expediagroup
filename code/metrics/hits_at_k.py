"""
    Copyright 2022 Expedia, Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
import tensorflow as tf
import numpy as np


class HitsAtK(tf.keras.metrics.Metric):
    """
    Calculates the Hits@K metric.
    """

    def __init__(self, name='HitsAtK', k=100, **kwargs):
        super(HitsAtK, self).__init__(name=name, **kwargs)
        self.hits_at_k = self.add_weight(name='hits_at_k', initializer='zeros')
        self.count_ = self.add_weight(name='counter', initializer='zeros')
        self.k = k

    def update_state(self, y_true, logits):
        logits = np.asarray(logits).astype(np.int)
        _, sorted_indices = tf.math.top_k(logits, k=self.k, sorted=True)

        y_true_r = tf.reshape(y_true, shape=[-1, 1])
        rel = tf.cast(tf.math.equal(tf.cast(y_true_r, tf.int32), sorted_indices), tf.int32)
        hits = tf.cast(tf.math.reduce_max(rel, axis=-1), dtype=tf.float32)

        self.hits_at_k.assign_add(tf.reduce_mean(hits))
        self.count_.assign_add(tf.constant(1.0))

    def result(self):
        return tf.math.divide(self.hits_at_k, self.count_)
