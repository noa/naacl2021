# Copyright 2021 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import tensorflow as tf

import math


class RankedListLoss:

    def __init__(self, margin, k, run_case1=True):
        super(RankedListLoss, self).__init__()
        self.margin = margin
        self.k = k
        self.run_case1 = run_case1

    def similarity_score(self, embeddings, labels):
        """ create similarity matrix via dot product and mask self """
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        batch_size = tf.shape(embeddings)[0]
        scores = tf.matmul(embeddings, tf.transpose(embeddings))

        eyes = tf.eye(batch_size, batch_size, dtype=tf.bool)
        labels = tf.reshape(labels, [-1, 1])
        y_ = tf.equal(labels, tf.transpose(labels))
        y_ = tf.cast(y_, dtype=tf.uint8)
        not_eye_mask = tf.math.logical_not(eyes)

        scores = tf.reshape(tf.boolean_mask(scores, not_eye_mask), (batch_size, -1))
        y_ = tf.reshape(tf.boolean_mask(y_, not_eye_mask), (batch_size, -1))
        # cast to add to scores
        y_ = tf.cast(y_, dtype=tf.float32)
        scores = tf.cast(scores, dtype=tf.float32)
        scores = tf.add(scores, self.margin * (1 - y_))
        return scores, y_


    def cond_op(self, i, loss, scores, y_bool, num_miss_pos, negative_mask):
        batch_size = tf.shape(scores)[0]
        return tf.less(i, batch_size)

    def while_op(self, i, loss, scores, y_bool, num_miss_pos, negative_mask):
        """ call op for tf.while_loop """
        row_mask = tf.gather(negative_mask, i)
        row_score = tf.gather(scores, i)
        row_miss = tf.gather(num_miss_pos, i)

        v, _ = tf.math.top_k(tf.boolean_mask(-row_score, row_mask), row_miss)
        v = v * -1
        loss = tf.add(loss, tf.reduce_sum(v))
        i = tf.add(i, 1)
        return i, loss, scores, y_bool, num_miss_pos, negative_mask

    def while_op_pos(self, i, loss, scores, y_bool, num_miss_neg, positive_mask):
        row_mask = tf.gather(positive_mask, i)
        row_score = tf.gather(scores, i)
        row_miss = tf.gather(num_miss_neg, i)


        v, _ = tf.math.top_k(tf.boolean_mask(row_score, row_mask), row_miss)
        loss = tf.add(loss, tf.reduce_sum(v))
        i = tf.add(i, 1)
        return i, loss, scores, y_bool, num_miss_neg, positive_mask

    def case1(self, scores, y):
        # case where n_pos < k
        value, index = tf.math.top_k(scores, self.k)
        batch_size = tf.shape(scores)[0]

        min_vals = tf.reduce_min(value, axis=1)
        min_vals = tf.reshape(min_vals, [batch_size, -1])
        min_vals = tf.broadcast_to(min_vals, tf.shape(scores))

        y_bool = tf.cast(y, dtype=tf.bool)
        loss_mask = scores < min_vals

        # separate positives
        positive_mask = tf.math.logical_and(y_bool, loss_mask)

        positive_miss_scores = tf.boolean_mask(scores, positive_mask)

        positive_loss = tf.reduce_sum(positive_miss_scores)

        loss_mask_n = scores >= min_vals
        negative_mask = tf.math.logical_and(tf.logical_not(y_bool), loss_mask_n)

        num_miss_pos = tf.reduce_sum(tf.cast(positive_mask, dtype=tf.int32), axis=1)
        negative_loss = tf.constant(0.0)
        i = tf.constant(0)
        res = tf.while_loop(self.cond_op, self.while_op,
                            [i, negative_loss, scores, y_bool, num_miss_pos, negative_mask])
        negative_loss = res[1]

        return negative_loss - positive_loss

    def case2(self, scores, y):
        # case where k >= n_pos
        value, index = tf.math.top_k(scores, self.k)
        batch_size = tf.shape(scores)[0]

        min_vals = tf.reduce_min(value, axis=1)
        min_vals = tf.reshape(min_vals, [batch_size, -1])
        min_vals = tf.broadcast_to(min_vals, tf.shape(scores))

        y_bool = tf.cast(y, dtype=tf.bool)
        loss_mask_n = scores >= min_vals
        negative_mask = tf.math.logical_and(tf.logical_not(y_bool), loss_mask_n)
        negative_loss = tf.reduce_sum(tf.boolean_mask(scores, negative_mask))

        loss_mask_p = scores < min_vals
        positive_mask = tf.math.logical_and(y_bool, loss_mask_p)

        num_miss_neg = tf.reduce_sum(tf.cast(negative_mask, dtype=tf.int32), axis=1)

        positive_loss = tf.constant(0.0)
        i = tf.constant(0)
        res = tf.while_loop(self.cond_op, self.while_op_pos,
                            [i, positive_loss, scores, y_bool, num_miss_neg, positive_mask])
        positive_loss = res[1]

        return negative_loss - positive_loss

    def __call__(self, embeddings, labels):
        batch_size = tf.shape(embeddings)[0]
        scores, y = self.similarity_score(embeddings, labels)

        # number of positives for each query
        num_pos = tf.reduce_sum(y, axis=1)
        # separate by case
        cond = num_pos < self.k

        case1_scores = tf.boolean_mask(scores, cond)
        case1_y = tf.boolean_mask(y, cond)

        case2_scores = tf.boolean_mask(scores, tf.logical_not(cond))
        case2_y = tf.boolean_mask(y, tf.logical_not(cond))

        if self.run_case1:
            loss = self.case1(case1_scores, case1_y) / tf.cast(batch_size,
                                                               dtype=tf.float32)
        else:
            loss = self.case2(case2_scores, case2_y)/tf.cast(batch_size, dtype=tf.float32)

        return loss
