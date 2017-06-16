# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import rpc_service_pb2 as rpc

vocabulary_size = 50000
batch_size = 256
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 100      # Number of negative examples to sample.


graph = tf.Graph()
tup = rpc.Tuple()
def _add_assign_and_placeholder(variable, tup, dtype):
    placeholder_node = tf.placeholder(tf.float32, variable.get_shape())
    assign_node = tf.assign(variable, placeholder_node)
    placeholder_indice_node = tf.placeholder(dtype)
    placeholder_gradient_node = tf.placeholder(tf.float32)
    scatter_sub_node = tf.scatter_sub(variable, placeholder_indice_node, placeholder_gradient_node)
    var_name = variable.name
    tup.map_names[var_name].variable_name = var_name
    tup.map_names[var_name].assign_name = assign_node.name
    #tup.map_names[var_name].assign_add_name = assign_add_node.name
    tup.map_names[var_name].placeholder_assign_name = placeholder_node.name
    tup.map_names[var_name].placeholder_indice_name = placeholder_indice_node.name
    tup.map_names[var_name].placeholder_gradient_name = placeholder_gradient_node.name
    tup.map_names[var_name].scatter_sub_name = scatter_sub_node.name[:-2]
    #tup.map_names[var_name].placeholder_assign_add_name = placeholder_assign_add_node.name

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  tup.batch_placeholder_name = train_inputs.name
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  tup.label_placeholder_name = train_labels.name
  #valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    print embeddings.name
    _add_assign_and_placeholder(embeddings, tup, tf.int32)
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    _add_assign_and_placeholder(nce_weights, tup, tf.int64)
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    _add_assign_and_placeholder(nce_biases, tup, tf.int64)

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
                     num_sampled, vocabulary_size))
  tup.loss_name = loss.name

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0)
  grads = optimizer.compute_gradients(loss)
  for grad_var in grads:
    tup.map_names[grad_var[1].name].gradient_name = grad_var[0].values.name
    tup.map_names[grad_var[1].name].gradient_index_name = grad_var[0].indices.name
    print grad_var[0].values.shape, grad_var[0].indices.shape, grad_var[0].indices.dtype
  training_op = optimizer.apply_gradients(grads)
  tup.training_op_name = training_op.name

  ## Compute the cosine similarity between minibatch examples and all embeddings.
  #norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  #normalized_embeddings = embeddings / norm
  #valid_embeddings = tf.nn.embedding_lookup(
  #    normalized_embeddings, valid_dataset)
  #similarity = tf.matmul(
  #    valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()
  tup.init_name = init.name

  tup.graph.CopyFrom(graph.as_graph_def())

  f = open("tuple_word2vec.pb", "wb")
  f.write(tup.SerializeToString())
  f.close()
