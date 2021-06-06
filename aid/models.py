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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import SeparableConv1D

from tensorflow_addons.layers import GroupNormalization

from aid.features import F
from aid.layers import SimpleAttentionEncoder
from aid.layers import LayerNormalizedProjection


class LinkModel(tf.keras.Model):
    def __init__(self, num_symbols=None, num_action_types=None,
                 padded_length=None, episode_len=16,
                 embedding_dim=512, num_layers=2, d_model=256,
                 num_heads=4, dff=256, dropout_rate=0.1,
                 subword_embed_dim=512, action_embed_dim=512,
                 filter_activation='relu', num_filters=256,
                 min_filter_width=2, max_filter_width=5,
                 final_activation='relu', use_gn=False, use_GLU=False,
                 use_attn_text_encoder=False,
                 use_separable_conv=False, time_encoding='one_hot',
                 **kwargs):
        super(LinkModel, self).__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.num_symbols = num_symbols
        self.num_action_types = num_action_types
        self.padded_length = padded_length
        self.episode_len = episode_len
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.subword_embed_dim = subword_embed_dim
        self.action_embed_dim = action_embed_dim
        self.min_filter_width = min_filter_width
        self.max_filter_width = max_filter_width
        self.num_filters = num_filters
        self.filter_activation = filter_activation
        self.final_activation = final_activation
        self.use_gn = use_gn
        self.use_GLU = use_GLU
        self.use_attn_text_encoder = use_attn_text_encoder
        self.time_encoding = time_encoding
        self.use_separable_conv = use_separable_conv

        self.subword_embedding = Embedding(self.num_symbols, self.subword_embed_dim,
                                           name='subword_embedding')
        self.action_embedding = Embedding(self.num_action_types,
                                          self.action_embed_dim,
                                          name='action_embedding')
        if self.use_attn_text_encoder:
            self.attn_text_encoder = SimpleAttentionEncoder(d_model=self.subword_embed_dim,
                                                            num_layers=self.num_layers)
        else:
            for width in range(self.min_filter_width, self.max_filter_width + 1):
                if self.use_separable_conv:
                    conv = SeparableConv1D(self.num_filters, width,
                                           depth_multiplier=1, activation=self.filter_activation)
                else:
                    conv = Conv1D(self.num_filters, width, activation=self.filter_activation)
                setattr(self, f'conv_{width}', conv)
                if self.use_gn:
                  setattr(self, f'norm_{width}', GroupNormalization())

        self.dense_1 = Dense(self.d_model)

        self.encoder = SimpleAttentionEncoder(d_model=self.d_model, num_layers=self.num_layers)

        self.mlp = LayerNormalizedProjection(self.embedding_dim,
                                             activation=self.final_activation)

    @tf.function
    def call(self, inputs, training=False):
        features = []

        # Extract text features
        net = inputs[F.SYMBOLS.value]
        batch_size = tf.shape(net)[0]
        episode_len = tf.shape(net)[1]
        net = tf.reshape(net, [-1, self.padded_length])
        swe = self.subword_embedding(net)
        if self.use_attn_text_encoder:
            net = self.attn_text_encoder(swe, training=training)
        else:
            fs = []
            for width in range(self.min_filter_width, self.max_filter_width + 1):
                layer = getattr(self, f'conv_{width}')
                net = layer(swe)
                if self.use_gn:
                    layer_norm = getattr(self, f'norm_{width}')
                    net = layer_norm(net)
                net = tf.reduce_max(net, axis=1, keepdims=False)
                fs.append(net)
            net = tf.concat(fs, axis=-1)
        feature_dim = net.get_shape()[-1]
        net = tf.reshape(net, [batch_size, episode_len, feature_dim])
        features.append(net)

        # Action embedding
        embedded_actions = self.action_embedding(inputs[F.ACTION_TYPE.value])
        features.append(embedded_actions)

        # Hour embedding
        hour = inputs[F.HOUR.value]
        features.append(tf.one_hot(hour, 24, dtype=tf.float32, name='hour_onehot'))

        lengths = inputs[F.NUM_POSTS.value]
        lengths = tf.reshape(lengths, [batch_size])
        mask = tf.sequence_mask(lengths, maxlen=episode_len)

        # Day embedding
        if F.DAY.value in inputs:
            features.append(tf.one_hot(inputs[F.DAY.value], 7, dtype=tf.float32, name='day_onehot'))

        net = tf.concat(features, axis=-1)
        net = self.dense_1(net)  # [batch_size, dim]
        net = self.encoder(net, training=training, mask=mask)
        net = self.mlp(net, training=training)
        return net


class LinkTextTimeModel(tf.keras.Model):
    def __init__(self, num_symbols=None, num_action_types=None,
                 padded_length=None, episode_len=16,
                 embedding_dim=512, num_layers=2, d_model=256,
                 num_heads=4, dff=256, dropout_rate=0.1,
                 subword_embed_dim=512, action_embed_dim=512,
                 filter_activation='relu', num_filters=256,
                 min_filter_width=2, max_filter_width=5,
                 final_activation='relu', use_gn=False, use_GLU=False,
                 use_attn_text_encoder=False,
                 use_separable_conv=False, time_encoding='one_hot',
                 **kwargs):
        super(LinkTextTimeModel, self).__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.num_symbols = num_symbols
        self.num_action_types = num_action_types
        self.padded_length = padded_length
        self.episode_len = episode_len
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.subword_embed_dim = subword_embed_dim
        self.action_embed_dim = action_embed_dim
        self.min_filter_width = min_filter_width
        self.max_filter_width = max_filter_width
        self.num_filters = num_filters
        self.filter_activation = filter_activation
        self.final_activation = final_activation
        self.use_gn = use_gn
        self.use_GLU = use_GLU
        self.use_attn_text_encoder = use_attn_text_encoder
        self.time_encoding = time_encoding
        self.use_separable_conv = use_separable_conv

        self.subword_embedding = Embedding(self.num_symbols, self.subword_embed_dim,
                                           name='subword_embedding')
        if self.use_attn_text_encoder:
            self.attn_text_encoder = SimpleAttentionEncoder(d_model=self.subword_embed_dim,
                                                            num_layers=self.num_layers)
        else:
            for width in range(self.min_filter_width, self.max_filter_width + 1):
                if self.use_separable_conv:
                    conv = SeparableConv1D(self.num_filters, width,
                                           depth_multiplier=1, activation=self.filter_activation)
                else:
                    conv = Conv1D(self.num_filters, width, activation=self.filter_activation)
                setattr(self, f'conv_{width}', conv)
                if self.use_gn:
                  setattr(self, f'norm_{width}', GroupNormalization())

        self.dense_1 = Dense(self.d_model)

        self.encoder = SimpleAttentionEncoder(d_model=self.d_model, num_layers=self.num_layers)

        self.mlp = LayerNormalizedProjection(self.embedding_dim,
                                             activation=self.final_activation)

    @tf.function
    def call(self, inputs, training=False):
        features = []
        # Extract text features
        net = inputs[F.SYMBOLS.value]
        batch_size = tf.shape(net)[0]
        episode_len = tf.shape(net)[1]
        net = tf.reshape(net, [-1, self.padded_length])
        swe = self.subword_embedding(net)
        if self.use_attn_text_encoder:
            net = self.attn_text_encoder(swe, training=training)
        else:
            fs = []
            for width in range(self.min_filter_width, self.max_filter_width + 1):
                layer = getattr(self, f'conv_{width}')
                net = layer(swe)
                if self.use_gn:
                  layer_norm = getattr(self, f'norm_{width}')
                  net = layer_norm(net)
                net = tf.reduce_max(net, axis=1, keepdims=False)
                fs.append(net)
            net = tf.concat(fs, axis=-1)
        feature_dim = net.get_shape()[-1]
        net = tf.reshape(net, [batch_size, episode_len, feature_dim])
        features.append(net)

        # No Action embedding

        # Hour embedding
        hour = inputs[F.HOUR.value]

        features.append(tf.one_hot(hour, 24, dtype=tf.float32, name='hour_onehot'))

        # Day embedding
        if F.DAY.value in inputs:
            features.append(tf.one_hot(inputs[F.DAY.value], 7, dtype=tf.float32, name='day_onehot'))

        lengths = inputs[F.NUM_POSTS.value]
        lengths = tf.reshape(lengths, [batch_size])
        mask = tf.sequence_mask(lengths, maxlen=episode_len)

        net = tf.concat(features, axis=-1)
        net = self.dense_1(net)  # [batch_size, dim]
        net = self.encoder(net, training=training, mask=mask)
        net = self.mlp(net, training=training)
        return net

