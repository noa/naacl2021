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

from absl import app
from absl import flags
from absl import logging
from functools import partial
from time import time
from tqdm import tqdm

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from tensorflow_addons.losses import TripletSemiHardLoss
from tensorflow_addons.losses import ContrastiveLoss

from aid.evaluation import author_id
from aid.evaluation import author_linking
from aid.features import F
from aid.features import FeatureConfig
from aid.losses import RankedListLoss
from aid.models import LinkModel
from aid.models import LinkTextTimeModel

import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'fit', ['fit', 'rank', 'link', 'benchmark_ds', 'embed'],
                  'Use `fit` to train model or `rank` to evaluate it')
flags.DEFINE_string('expt_dir', None, 'Experiment directory')
flags.DEFINE_string('results_filename', 'results.txt', 'Written as expt_dir/results_filename')
flags.DEFINE_integer('fit_verbosity', 1, 'Use `1` for local jobs and `2` for grid jobs')
flags.DEFINE_integer('save_freq', 50000, 'Number of steps between checkpoints')
flags.DEFINE_string('train_tfrecord_path', None, 'Path to train TFRecords')
flags.DEFINE_string('valid_tfrecord_path', None, 'Path to validation TFRecords')
flags.DEFINE_string('monitor', 'MRR', 'Metric to monitor')
flags.DEFINE_integer('num_cpu', 4, 'Number of CPU processes')
flags.DEFINE_integer('num_classes', None, 'This is usually the number of authors')
flags.DEFINE_integer('filter_author_num', None, 'Filter authors down to this many')
flags.DEFINE_integer('episode_len', 16, 'Episode length')
flags.DEFINE_integer('samples_per_class', 4, 'Number of samples for each author history')
flags.DEFINE_integer('embedding_dim', 512, 'Embedding dimensionality')
flags.DEFINE_integer('num_epochs', 20, 'Number of training epochs')
flags.DEFINE_integer('steps_per_epoch', 10000, 'Steps per epoch')
flags.DEFINE_integer('valid_steps', 100, 'Number of validation steps')
flags.DEFINE_integer('num_queries', 25000, 'Number of ranking queries')
flags.DEFINE_string('optimizer', 'sgd', 'Optimizer')
flags.DEFINE_boolean('use_lookahead', False, 'Use lookahead optimizer')
flags.DEFINE_float('learning_rate', 0.05, 'Learning rates')
flags.DEFINE_integer('first_decay_steps', 10000, 'First decay steps for restarts')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay scale')
flags.DEFINE_string('schedule', 'piecewise', 'Learning rate schedule')
flags.DEFINE_float('momentum', 0.9, 'Momentum')
flags.DEFINE_boolean('nesterov', False, 'Use Nesterov momentum')
flags.DEFINE_float('grad_norm_clip', 1., 'Clip the norm of gradients to this value')
flags.DEFINE_integer('batch_size', 128, 'Mini-batch size for SGD')
flags.DEFINE_enum('loss', 'triplet', ['ranked_list', 'contrastive', 'triplet'],
                  'Surrogate metric learning objective')
flags.DEFINE_float('margin', 0.5, 'Triplet margin')
flags.DEFINE_string('final_activation', 'relu', 'Final activation')
flags.DEFINE_integer('num_filters', 256, 'Number of filters')
flags.DEFINE_integer('min_filter_width', 2, 'Smallest filter size')
flags.DEFINE_integer('max_filter_width', 5, 'Largest filter size')
flags.DEFINE_string('filter_activation', 'relu', 'Nonlinearity after feature')
flags.DEFINE_integer('num_layers', 2, 'Number of Transformer encoder layers')
flags.DEFINE_integer('d_model', 256, 'Transformer layer size')
flags.DEFINE_integer('num_heads', 4, 'Number of attention heads')
flags.DEFINE_integer('dff', 256, 'Size of feedforward layers in encoder')
flags.DEFINE_integer('log_steps', 1000, 'Steps at which to log progress')
flags.DEFINE_float('dropout_rate', 0.1, 'Rate of dropout')
flags.DEFINE_integer('subword_embed_dim', 512, 'Size of subword embedding')
flags.DEFINE_integer('action_embed_dim', 512, 'Size of action type embedding')
flags.DEFINE_string('expt_config_path', 'results', 'Model and log directory')
flags.DEFINE_integer('num_parallel_readers', 4, 'Number of files to read in parallel')
flags.DEFINE_integer('shuffle_seed', 42, 'Seed for data shuffle')
flags.DEFINE_integer('shuffle_buffer_size', 2 ** 13, 'Size of shufle buffer')
flags.DEFINE_string('distance', 'cosine', 'How to compare embeddings')
flags.DEFINE_string('model_type', 'IurMini', 'Type of aid model to use')
flags.DEFINE_integer('k', 5, 'K value to optimize for Prec@K loss')
flags.DEFINE_float('k_margin', 0.1, 'Margin used in Prec@K loss')
flags.DEFINE_boolean('mixed_precision', False, 'Use mixed percision')
flags.DEFINE_boolean('group_normalization', False, 'Group normalize cnn layers')
flags.DEFINE_boolean('attn_text_encoder', False, 'Use attention for text encoding')
flags.DEFINE_boolean('separable_convolutions', False, 'Use separable convs in text encoder')
flags.DEFINE_string('time_encoding', 'one_hot', "Type of time encoding to use: either [one_hot, cyclical]")
flags.DEFINE_boolean('freeze_batch_norm_vars', False, 'Freeze batch normalization weights')
flags.DEFINE_float('proxy_lr_multiplier', 100.0, 'Multiplier to apply to proxies')
flags.DEFINE_string('training_records', '',
                    "Training data as sharded tfrecords.")
flags.DEFINE_boolean('time_based_checkpointing', False, 'Store new checkpoint every hour regardless of performance')
flags.DEFINE_float('checkpoint_interval', 1.0, 'How often to save a checkpoint, in hours')
flags.DEFINE_integer('piecewise_decay_steps_1', 80000, 'The first decay point for piecewise decay')
flags.DEFINE_integer('piecewise_decay_steps_2', 140000, 'The second decay point for piecewise decay')
flags.DEFINE_integer('piecewise_decay_steps_3', 200000, 'The final decay point for piecewise decay')
flags.DEFINE_boolean('warm_start_model', False, 'Warm start from embedding already in expt_dir')
flags.DEFINE_integer('min_episode_len', 16, 'Minimum episode length (for training on variable length episodes)')
flags.DEFINE_integer('min_val_len', 16, 'Minimum validation episode length')
flags.DEFINE_integer('max_val_len', 16, 'Maximum validation episode length')
flags.DEFINE_string('linking_queries', None, 'Queries for linking experiment')
flags.DEFINE_string('linking_targets', None, 'targets for linking experiment')
flags.DEFINE_string('linking_test_targets', None, 'targets for linking experiment')
flags.DEFINE_boolean('run_linking', False, 'Whether to run linking or ranking validation')
flags.DEFINE_string('output_embed_path', 'embed.npy', 'Output path for embed mode')
flags.DEFINE_string('index_array_path', None,
                    'Path to numpy array containing starting index and length for episodes for each author to embed')
flags.DEFINE_boolean('random_episodes', False, 'Whether to choose episodes randomly (instead of contiguously)')
flags.DEFINE_boolean('random_window_sample', False, 'If set to true, a random text window will be sampled per document')


def get_flagfile():
  return os.path.join(FLAGS.expt_dir, 'flags.cfg')


def get_ckpt_dir():
  return os.path.join(FLAGS.expt_dir, 'checkpoints')


def get_export_dir():
  return os.path.join(FLAGS.expt_dir, 'embedding')


def build_episode_embedding(config):
  if 'Full' in FLAGS.model_type:
    return LinkModel(num_symbols=config.num_symbols,
                     num_action_types=config.num_action_types,
                     padded_length=config.padded_length,
                     embedding_dim=FLAGS.embedding_dim,
                     episode_len=FLAGS.episode_len,
                     features=FLAGS.features,
                     num_layers=FLAGS.num_layers,
                     d_model=FLAGS.d_model,
                     num_heads=FLAGS.num_heads,
                     dff=FLAGS.dff,
                     dropout_rate=FLAGS.dropout_rate,
                     subword_embed_dim=FLAGS.subword_embed_dim,
                     action_embed_dim=FLAGS.action_embed_dim,
                     filter_activation=FLAGS.filter_activation,
                     final_activation=FLAGS.final_activation,
                     num_filters=FLAGS.num_filters,
                     min_filter_width=FLAGS.min_filter_width,
                     max_filter_width=FLAGS.max_filter_width,
                     use_gn=FLAGS.group_normalization,
                     use_attn_text_encoder=FLAGS.attn_text_encoder,
                     use_separable_conv=FLAGS.separable_convolutions,
                     time_encoding=FLAGS.time_encoding)
  elif 'TextTime' in FLAGS.model_type:
    return LinkTextTimeModel(num_symbols=config.num_symbols,
                             num_action_types=config.num_action_types,
                             padded_length=config.padded_length,
                             embedding_dim=FLAGS.embedding_dim,
                             episode_len=FLAGS.episode_len,
                             features=FLAGS.features,
                             num_layers=FLAGS.num_layers,
                             d_model=FLAGS.d_model,
                             num_heads=FLAGS.num_heads,
                             dff=FLAGS.dff,
                             dropout_rate=FLAGS.dropout_rate,
                             subword_embed_dim=FLAGS.subword_embed_dim,
                             action_embed_dim=FLAGS.action_embed_dim,
                             filter_activation=FLAGS.filter_activation,
                             final_activation=FLAGS.final_activation,
                             num_filters=FLAGS.num_filters,
                             min_filter_width=FLAGS.min_filter_width,
                             max_filter_width=FLAGS.max_filter_width,
                             use_gn=FLAGS.group_normalization,
                             use_attn_text_encoder=FLAGS.attn_text_encoder,
                             use_separable_conv=FLAGS.separable_convolutions,
                             time_encoding=FLAGS.time_encoding)


class Model(tf.keras.Model):
  def __init__(self, config, **kwargs):
    super(Model, self).__init__(**kwargs)
    self.embedding = build_episode_embedding(config)

  @tf.function
  def call(self, inputs, training=False):
    v = self.embedding(inputs, training=training)

    if 'ranked_list' in FLAGS.loss:
      logits = v
    elif 'triplet' in FLAGS.loss or 'contrastive' in FLAGS.loss:
      logits = tf.nn.l2_normalize(v, axis=1)

    if FLAGS.mixed_precision:
      logits = tf.cast(logits, dtype=tf.float32)
    return logits


def sample_random_episode(dataset, config,
                          min_episode_length=16,
                          max_episode_length=16,
                          repeat=1):
  assert repeat > 0

  def sample_episode(features, label):
    if min_episode_length == max_episode_length:
      length = min_episode_length
    else:
      # sample from beta distrib
      length = np.ceil((max_episode_length - min_episode_length) * np.random.beta(3, 1)).astype(
        np.int64) + min_episode_length
    num_action = features[F.NUM_POSTS.value]
    new_features = {}

    if FLAGS.random_episodes:
      indices = tf.random.shuffle(tf.range(num_action))[:length]
      for key in config.sequence_features:
        new_features[key.value] = tf.gather(features[key.value], indices)
    else:
      maxval = num_action - length + 1
      start_index = tf.reshape(tf.random.uniform([1], minval=0, maxval=maxval,
                                                 dtype=tf.dtypes.int64), [])
      end_index = start_index + length
      for key in config.sequence_features:
        new_features[key.value] = features[key.value][start_index:end_index]

    for key in config.context_features:
      new_features[key.value] = features[key.value]
    new_features[F.NUM_POSTS.value] = length  # length of episode
    return new_features, label

  if repeat < 2:
    return dataset.map(
      sample_episode,
      num_parallel_calls=1)
  else:
    def repeat_sample_episode(features, label):
      xs = {}
      ys = []
      for _ in range(repeat):
        x, y = sample_episode(features, label)
        for k, v in x.items():
          if k in xs:
            xs[k].append(v)
          else:
            xs[k] = [v]
        ys.append(y)
      for k, v in xs.items():
        xs[k] = tf.ragged.stack(v, axis=0)
        if type(xs[k]) is tf.RaggedTensor:
          xs[k] = xs[k].to_tensor()  # convert ragged to uniform because padded_batch won't accept ragged
      return xs, tf.stack(ys, axis=0)

    ds = dataset.map(repeat_sample_episode, num_parallel_calls=1)
    return ds.unbatch()


def build_dataset(file_pattern, config, num_epochs=None, shuffle=True,
                  random_episode=True, take=None, samples_per_class=1,
                  filter_num_posts=None, filter_authors=False, val=False):
  filenames = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
  if shuffle:
    ds = filenames.interleave(
      tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(FLAGS.shuffle_buffer_size, reshuffle_each_iteration=True)
  else:
    ds = tf.data.TFRecordDataset(filenames)

  if take and take > 0:
    ds = ds.take(take)
  ds = ds.repeat(num_epochs)
  ds = ds.map(config.parse_single_example_fn,
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if random_episode:
    min_episode_len = FLAGS.min_episode_len
    max_episode_len = FLAGS.episode_len
    if val:
      min_episode_len = FLAGS.min_val_len
      max_episode_len = FLAGS.max_val_len

    ds = sample_random_episode(
      ds, config,
      min_episode_length=min_episode_len,
      max_episode_length=max_episode_len,
      repeat=samples_per_class)

  if filter_authors and FLAGS.filter_author_num is not None:
    def filter_authors(features, label):
      author_id = features[F.AUTHOR_ID.value]
      return author_id < FLAGS.filter_author_num

    ds = ds.filter(filter_authors)

  if filter_num_posts:
    assert filter_num_posts > 1

    def filter_posts(features, label):
      num_posts = features[F.NUM_POSTS.value]
      return num_posts > filter_num_posts

    ds = ds.filter(filter_posts)

  ds = ds.padded_batch(FLAGS.batch_size,
                       padded_shapes=(
                       {'action_type': [None], 'lens': [None], 'syms': ([None, None]), 'hour': [None], 'author_id': (),
                        'num_posts': ()}, ()))

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  # Make sure autotune doesn't steal unallocated CPUs on shared compute nodes
  options = tf.data.Options()
  options.experimental_optimization.autotune_cpu_budget = FLAGS.num_cpu
  ds = ds.with_options(options)

  return ds


def build_pairs_dataset(file_pattern, config, num_epochs=None):
  files = tf.data.Dataset.list_files(file_pattern, shuffle=False)
  ds = tf.data.TFRecordDataset(files)
  scrambledFiles = tf.data.Dataset.list_files(file_pattern, shuffle=True)
  dss = scrambledFiles.interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers)

  ds = ds.map(config.parse_single_example_fn, num_parallel_calls=1)
  dss = dss.map(config.parse_single_example_fn, num_parallel_calls=FLAGS.num_cpu)

  ds1 = sample_random_episode(ds, config)
  ds2 = sample_random_episode(ds, config)
  one = tf.data.Dataset.from_tensors(1).repeat(FLAGS.num_classes)
  positive = tf.data.Dataset.zip((ds1, ds2, one))

  dss = dss.shuffle(FLAGS.shuffle_buffer_size, reshuffle_each_iteration=False)
  ds3 = sample_random_episode(dss, config)
  zero = tf.data.Dataset.from_tensors(0).repeat(FLAGS.num_classes)
  negative = tf.data.Dataset.zip((ds1, ds3, zero))

  ds = positive.concatenate(negative)
  ds = ds.shuffle(2 * FLAGS.num_classes)

  ds = ds.repeat(num_epochs)
  ds = ds.batch(FLAGS.batch_size)
  ds = ds.prefetch(1)
  return ds


def build_sliced_embedding(model, file_pattern, config):
  filenames = tf.data.Dataset.list_files(file_pattern)
  ds = tf.data.TFRecordDataset(filenames)
  ds = ds.map(config.parse_single_example_fn, num_parallel_calls=FLAGS.num_cpu)
  if FLAGS.index_array_path:
    # ASSUMING that the slices is ordered in the same way as the dataset
    def sliced_episodes_map(dataset, slices):
      features = dataset[0]
      label = dataset[1]
      start_index = tf.reshape(slices[0], [])
      episode_length = tf.reshape(slices[1], [])
      end_index = start_index + episode_length

      new_features = {}
      for key in config.sequence_features:
        new_features[key.value] = features[key.value][start_index:end_index]
      for key in config.context_features:
        new_features[key.value] = features[key.value]
      new_features[F.NUM_POSTS.value] = episode_length
      return new_features, label

    slices = tf.data.Dataset.from_tensor_slices(np.load(FLAGS.index_array_path))
    zipped_dataset = tf.data.Dataset.zip((ds, slices))
    ds = zipped_dataset.map(sliced_episodes_map, num_parallel_calls=1)

  ds = ds.padded_batch(FLAGS.batch_size, padded_shapes=(
  {'action_type': [None], 'lens': [None], 'syms': ([None, None]), 'hour': [None], 'author_id': (), 'num_posts': ()},
  ()))
  ds = ds.prefetch(1)
  return ds


def embed(model, file_pattern, config):
  dataset = build_sliced_embedding(model, file_pattern, config)
  embeddings = []
  authors = []
  for batch in dataset:
    features, labels = batch
    e = model.predict(features, batch_size=FLAGS.batch_size)
    embeddings.append(e)
  return np.vstack(embeddings)


def embedding_and_labels(model, file_pattern, config, take=None,
                         random_episode=True, val=True):
  # if we're running the baseline here, we sample multiple times and will later average the embeddingsf
  samples = 1
  dataset = build_dataset(file_pattern,
                          config,
                          num_epochs=1,
                          shuffle=False,
                          take=take,
                          val=val,
                          samples_per_class=samples)
  embeddings = []
  authors = []
  for batch in dataset:
    features, labels = batch
    if FLAGS.mode == 'fit':
      e = model(features, training=False)
    else:
      e = model.predict(features, batch_size=FLAGS.batch_size)
    embeddings.append(e)
    authors.append(labels)
  return np.vstack(embeddings), np.concatenate(authors)


def rank(model, config, checkpoint=None):
  logging.info(f"Computing queries from {FLAGS.train_tfrecord_path}")
  query_vectors, query_labels = embedding_and_labels(
    model, FLAGS.train_tfrecord_path, config, take=FLAGS.num_queries)
  logging.info(f"{query_vectors.shape[0]} queries")
  logging.info(f"Computing targets from {FLAGS.valid_tfrecord_path}")
  target_vectors, target_labels = embedding_and_labels(
    model, FLAGS.valid_tfrecord_path, config)
  logging.info(f"{target_vectors.shape[0]} targets")
  logging.info(f"Performing ranking evaluation")
  metrics = author_id(
    query_vectors, query_labels, target_vectors, target_labels, metric=FLAGS.distance,
    n_jobs=FLAGS.num_cpu)
  return metrics


def pointwise_average_embeddings(embeddings, labels):
  # pointwise average all embeddings of the same label
  averaged_embeddings = []
  averaged_labels = []
  all_labels = np.unique(labels)
  for l in all_labels:
    averaged_labels.append(l)
    idx = np.where(labels == l)[0]
    averaged_embeddings.append(np.average(embeddings[idx], axis=0))

  return np.vstack(averaged_embeddings), np.concatenate(averaged_labels)


def link(model, config):
  logging.info(f"Computing queries from {FLAGS.linking_queries}")
  query_vectors, query_labels = embedding_and_labels(model, FLAGS.linking_queries, config, val=False)
  logging.info(f"{query_vectors.shape[0]} queries")
  logging.info(f"Computing targets from {FLAGS.linking_targets}")
  target_vectors, target_labels = embedding_and_labels(model, FLAGS.linking_targets, config, val=True)
  logging.info(f"{target_vectors.shape[0]} targets")

  logging.info(f"Computing test targets from {FLAGS.linking_test_targets}")
  target_test_vectors, target_test_labels = embedding_and_labels(model, FLAGS.linking_test_targets, config)
  logging.info(f"{target_test_vectors.shape[0]} targets")

  logging.info(f"Performing linking evaluation")

  metrics = author_linking(query_vectors, query_labels,
                           target_vectors, target_labels,
                           target_test_vectors, target_test_labels)

  return metrics


def get_lr_schedule():
  if FLAGS.schedule == 'constant':
    return PiecewiseConstantDecay([0], [FLAGS.learning_rate, FLAGS.learning_rate])
  elif FLAGS.schedule == 'piecewise':
    steps = [FLAGS.piecewise_decay_steps_1, FLAGS.piecewise_decay_steps_2, FLAGS.piecewise_decay_steps_3]
    lr = FLAGS.learning_rate
    return PiecewiseConstantDecay(
      steps,
      [lr, lr / 10., lr / 100., lr / 1000.])
  elif FLAGS.schedule == 'cosine_decay_restarts':
    return tf.keras.experimental.CosineDecayRestarts(
      FLAGS.learning_rate, FLAGS.first_decay_steps)
  else:
    raise ValueError(FLAGS.schedule)


def get_optimizer():
  if FLAGS.optimizer == 'sgd':
    optimizer = partial(SGD, momentum=FLAGS.momentum,
                        nesterov=FLAGS.nesterov)
  elif FLAGS.optimizer == 'adam':
    optimizer = partial(Adam)
  elif FLAGS.optimizer == 'adamw':
    optimizer = partial(tfa.optimizers.AdamW)
  elif FLAGS.optimizer == 'adamr':
    optimizer = partial(tfa.optimizers.RectifiedAdam)
  else:
    raise ValueError(FLAGS.optimizer)
  return optimizer


def build_optimizer(step=None):
  schedule = get_lr_schedule()
  optimizer = get_optimizer()

  if FLAGS.optimizer in ['adamw']:
    if FLAGS.schedule == 'constant':
      opt = optimizer(learning_rate=schedule,
                      weight_decay=FLAGS.weight_decay)
    else:
      raise NotImplementedError
  else:
    opt = optimizer(learning_rate=schedule)
  if FLAGS.use_lookahead:
    opt = tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)
  if FLAGS.mixed_precision:
    opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')
  return opt


def load_embedding(config, export_path):
  embedding = build_episode_embedding(config)

  # This initializes the variables associated with the embedding
  ds = build_dataset(FLAGS.train_tfrecord_path, config, shuffle=False, val=True)
  for x, _ in ds.take(1):
    e = embedding(x)
    logging.info(f"Output shape: {e.shape}")

  # Load model weights
  embedding.load_weights(export_path)

  return embedding


def sanity_check_saved_model(model, config, export_path):
  if FLAGS.mixed_precision:
    policy = tf.keras.mixed_precision.experimental.Policy('float32')
    mixed_precision.set_policy(policy)
  logging.info(f"Loading weights from {export_path}...")
  new_model = load_embedding(config, export_path)
  logging.info(f"Weights loaded")
  ds = build_dataset(FLAGS.valid_tfrecord_path, config, shuffle=False, val=True)
  for x, _ in ds.take(1):
    predictions = model(x, training=False)
    new_predictions = new_model.predict(x)
  np.testing.assert_allclose(predictions, new_predictions, rtol=1e-4, atol=1e-4)
  logging.info("Serialization worked!")


def attempt_model_logging(model):
  model.summary(print_fn=logging.info)
  try:
    model.embedding.summary(print_fn=logging.info)
  except:
    logging.info("Failed to log model embedding summary")
  try:
    model.projection.summary(print_fn=logging.info)
  except:
    logging.info("Failed to log model projection layer summary")


def custom_fit(model, config):
  total_num_steps = FLAGS.num_epochs * FLAGS.steps_per_epoch
  if 'ranked_list' in FLAGS.loss:
    if FLAGS.samples_per_class < FLAGS.k:
      compute_loss = RankedListLoss(FLAGS.k_margin, FLAGS.k, run_case1=True)
    else:
      compute_loss = RankedListLoss(FLAGS.k_margin, FLAGS.k, run_case1=False)
  elif 'triplet' in FLAGS.loss:
    compute_loss = TripletSemiHardLoss(margin=FLAGS.margin)
  elif 'contrastive' in FLAGS.loss:
    compute_loss = ContrastiveLoss()
  else:
    raise ValueError("Unknown loss function")

  train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
  optimizer = build_optimizer()


  @tf.function
  def train_one_step(model, optimizer, inputs, targets):
    # By default `GradientTape` will automatically watch any trainable
    # variables that are accessed inside the context
    with tf.GradientTape() as tape:

      if 'contrastive' in FLAGS.loss:
        inputs1, inputs2 = inputs
        logits1 = model(inputs1, training=True)
        logits2 = model(inputs2, training=True)
        m = tf.squeeze(targets)
        d = tf.linalg.norm(logits1 - logits2, axis=1)
        loss_value = tf.reduce_mean(tfa.losses.contrastive_loss(m, d, margin=1.0))
      else:
        logits = model(inputs, training=True)
        if 'ranked_list' in FLAGS.loss:
          loss_value = compute_loss(logits, targets)
        else:
          loss_value = compute_loss(y_true=targets, y_pred=logits)
      if FLAGS.mixed_precision:
        scaled_loss_value = optimizer.get_scaled_loss(loss_value)

    if FLAGS.freeze_batch_norm_vars:
      vars_to_optimize = []
      for var in model.trainable_variables:
        if FLAGS.freeze_batch_norm_vars and 'BatchNorm' in var.name:
          continue
        vars_to_optimize.append(var)
    else:
      vars_to_optimize = model.trainable_variables

    if FLAGS.mixed_precision:
      grads = tape.gradient(scaled_loss_value, vars_to_optimize)
      grads = optimizer.get_unscaled_gradients(grads)
    else:
      grads = tape.gradient(loss_value, vars_to_optimize)
    train_loss(loss_value)
    if FLAGS.grad_norm_clip:
      grads, _ = tf.clip_by_global_norm(grads, FLAGS.grad_norm_clip)

    optimizer.apply_gradients(zip(grads, vars_to_optimize))
    return loss_value

  def train(model):
    if FLAGS.training_records == '':
      true_query_path = FLAGS.train_tfrecord_path
    else:
      true_query_path = FLAGS.training_records

    train_ds = build_dataset(true_query_path, config,
                             samples_per_class=FLAGS.samples_per_class,
                             shuffle=True, random_episode=True,
                             filter_authors=True)

    train_ds = train_ds.take(total_num_steps)

    best_score = -1
    step = 0
    epoch = 0
    start_time = time()

    ckpt_num = 0
    logged_summary = False
    disable_progress = FLAGS.fit_verbosity == 2

    train_log_dir = FLAGS.expt_dir + '/train_log'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    if FLAGS.mixed_precision:
      policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
      mixed_precision.set_policy(policy)

    for X in tqdm(train_ds,
                  total=total_num_steps, disable=disable_progress):

      if 'contrastive' in FLAGS.loss:
        x1, x2, m = X
        loss = train_one_step(model, optimizer, (x1, x2), m)

      else:
        x, y = X

        loss = train_one_step(model, optimizer, x, y)
      step += 1

      if not logged_summary:
        logging.info(len(model.trainable_variables))
        attempt_model_logging(model)
        logged_summary = True

      if step % FLAGS.log_steps == 0:
        logging.info((f"[Step {step} of {total_num_steps}] loss {loss.numpy():.2f}, "
                      f"lr {optimizer.lr(step).numpy():.3f}, "
                      f"best {FLAGS.monitor} {best_score:.3f}"))
      if step % 200 == 0:
        with train_summary_writer.as_default():
          tf.summary.scalar('loss', train_loss.result(), step=step)
          tf.summary.scalar('learning_rate',
                            optimizer.lr(step).numpy(), step=step)

      if FLAGS.time_based_checkpointing:
        export_path = get_export_dir()
        curr_time = time()
        if (curr_time - start_time) / 3600. > FLAGS.checkpoint_interval:
          full_embedding_path = export_path + '_' + str(ckpt_num)
          logging.info(f"Exporting model to {full_embedding_path}")
          model.embedding.save_weights(full_embedding_path, save_format='tf')
          start_time = curr_time
          ckpt_num += 1

      if step > 0 and step % FLAGS.steps_per_epoch == 0:
        epoch += 1
        logging.info(f"[Epoch {epoch} of {FLAGS.num_epochs}]")

        metrics = rank(model.embedding, config)
        if FLAGS.run_linking:
          metrics_link = link(model.embedding, config)
          metrics = {**metrics, **metrics_link}

        logging.info(f"Ranking metrics:")
        for name in sorted(metrics.keys()):
          score = metrics[name]
          logging.info(f"{name} {score:.3f}")
        with train_summary_writer.as_default():
          tf.summary.scalar('recall@8', metrics['recall@8'], step=step)
          tf.summary.scalar('median_rank', metrics['median_rank'], step=step)

        if metrics[FLAGS.monitor] > best_score:
          best_score = metrics[FLAGS.monitor]
          export_path = get_export_dir()
          logging.info(f"Exporting best model to {export_path}")
          model.embedding.save_weights(export_path, save_format='tf')
          if epoch == 1:
            logging.info("Checking weight serialization...")
            sanity_check_saved_model(model.embedding, config, export_path)

  logging.info("Training!")
  train(model)


def fit(config):
  flags.mark_flags_as_required(['num_classes',
                                'train_tfrecord_path',
                                'valid_tfrecord_path',
                                'expt_dir'])
  total_num_steps = FLAGS.num_epochs * FLAGS.steps_per_epoch
  logging.info(f"Training for {total_num_steps} steps total")
  checkpoint_file = "weights.{epoch:02d}.ckpt"
  checkpoint_path = get_ckpt_dir() + "/" + checkpoint_file
  checkpoint_dir = os.path.dirname(checkpoint_path)
  logging.info(f"Checkpoint directory: {checkpoint_dir}")

  model = Model(config)

  if FLAGS.warm_start_model:
    logging.info('Warm starting model from:', get_export_dir())
    model.embedding.load_weights(get_export_dir())

  custom_fit(model, config)


def handle_flags(argv):
  key_flags = FLAGS.get_key_flags_for_module(argv[0])
  s = '\n'.join(f.serialize() for f in key_flags)
  logging.info(f'fit.py flags:\n{s}')
  if FLAGS.mode == 'fit':
    flagfile = get_flagfile()
    with open(flagfile, 'w') as fh:
      logging.info(f"Writing flags to {flagfile}")
      fh.write(s)


def main(argv):
  logging.info(f"Limiting to {FLAGS.num_cpu} CPU")
  tf.config.threading.set_inter_op_parallelism_threads(FLAGS.num_cpu)
  tf.config.threading.set_intra_op_parallelism_threads(FLAGS.num_cpu)

  config = FeatureConfig.from_json(FLAGS.expt_config_path)
  logging.info(config)

  if FLAGS.min_episode_len > FLAGS.episode_len:
    raise ValueError("min_episode_len must be less than or equal to episode_len")

  if FLAGS.min_val_len > FLAGS.max_val_len:
    raise ValueError("min_val_len must be less than or equal to max_val_len")

  if FLAGS.mode == 'benchmark_ds':
    import tensorflow_datasets as tfds
    ds = build_dataset(FLAGS.train_tfrecord_path, config,
                       samples_per_class=FLAGS.samples_per_class)
    tfds.core.benchmark(ds.take(1000), batch_size=FLAGS.batch_size)

  if FLAGS.mode == 'fit':
    handle_flags(argv)
    fit(config)

  if FLAGS.mode == 'rank':
    flags.mark_flags_as_required(
        ['train_tfrecord_path', 'expt_dir', 'valid_tfrecord_path'])
    export_dir = get_export_dir()
    logging.info(f"Loading embedding from {export_dir}")
    embedding = load_embedding(config, export_dir)
    metrics = rank(embedding, config)
    results_file = os.path.join(FLAGS.expt_dir, FLAGS.results_filename)
    logging.info(f"Results will be written to {results_file}")
    results = "\n".join([f"{k} {v}" for k, v in metrics.items()])
    logging.info(results)

    with open(results_file, 'w') as fh:
      fh.write(results)

  if FLAGS.mode == 'link':
    export_dir = get_export_dir()
    logging.info(f"Loading embedding from {export_dir}")
    embedding = load_embedding(config, export_dir)
    metrics = link(embedding, config)
    results = "\n".join([f"{k} {v}" for k, v in metrics.items()])
    logging.info(results)

  if FLAGS.mode == 'embed':
    export_dir = get_export_dir()
    logging.info(f"Loading embedding from {export_dir}")
    embedding = load_embedding(config, export_dir)
    logging.info(f"Producing embeddings of data: {FLAGS.train_tfrecord_path}")
    vectors = embed(embedding, FLAGS.train_tfrecord_path, config)
    shape = vectors.shape
    logging.info((f"Writing {shape[0]} embeddings of size {shape[1]}"
                  f" to {FLAGS.output_embed_path}"))
    np.save(FLAGS.output_embed_path, vectors)


if __name__ == '__main__':
  app.run(main)
