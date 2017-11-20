
from __future__ import absolute_import
from __future__ import division

import os
import sys
import tensorflow as tf
import rpc_service_pb2 as rpc

NUM_CLASSES = 10
batch_size = int(sys.argv[1])

def _variable_on_cpu(name, shape, initializer, tup):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    placeholder_assign_node = tf.placeholder(dtype, shape=shape)
    assign_node = tf.assign(var, placeholder_assign_node)
    var_name = var.name
    tup.map_names[var_name].variable_name = var_name
    tup.map_names[var_name].assign_name = assign_node.name
    tup.map_names[var_name].placeholder_assign_name = placeholder_assign_node.name

  return var


def _variable_with_weight_decay(name, shape, stddev, wd, tup):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype), tup)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def inference(images, tup):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1, 5*5*3*64 = 4.8k, 4.8*4=19.2kB
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0, 
                                         tup=tup)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0), tup)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2, 5*5*64*64 = 102.4k, 102.4k*4=409.6kB
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0,
                                         tup=tup)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1), tup)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  factor = 3 * 5
  a = 0.00
  # local3, 7*7*64*384*3 = 3.612672M 3.612672M*4 = 14.450688MB
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384*factor],
                                          stddev=0.02, wd=a, tup=tup)
    biases = _variable_on_cpu('biases', [384*factor], tf.constant_initializer(0.1), tup)
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  dim2 = 1920
  # local4, 384*3*1920 = 2.211840M, 2.211840*4 = 8.847360MB
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384*factor, dim2],
                                          stddev=0.04, wd=a, tup=tup)
    biases = _variable_on_cpu('biases', [dim2], tf.constant_initializer(0.1), tup)
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)


  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency. 1920*10=19.2k, 19.2k*4=76.8k
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [dim2, NUM_CLASSES],
                                          stddev=1.0/dim2, wd=0.0, tup=tup)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0), tup)
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)


  return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

with tf.Session() as sess:
  # Build a Graph that computes the logits predictions from the
  # inference model.
  tup = rpc.Tuple()
  images = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 3])
  labels = tf.placeholder(tf.int32, shape=[batch_size])
  tup.batch_placeholder_name = images.name
  tup.label_placeholder_name = labels.name
  logits = inference(images, tup)

  # Calculate loss.
  losses = loss(logits, labels)

  opt = tf.train.GradientDescentOptimizer(learning_rate=0.2)
  grads = opt.compute_gradients(losses)
  for grad_var in grads:
    print grad_var[1].name
    print grad_var[0].name
    tup.map_names[grad_var[1].name].gradient_name = grad_var[0].name
  
  training_op = opt.apply_gradients(grads)
  tup.training_op_name = training_op.name

  # Create a saver.
  saver = tf.train.Saver(tf.all_variables())

  # Build an initialization operation to run below.
  init = tf.global_variables_initializer()

  tup.lr = 0.1
  tup.graph.CopyFrom(sess.graph_def)
  tup.loss_name = losses.name
  tup.init_name = init.name

  f = open("tuple_gradient_descent.pb", "wb")
  f.write(tup.SerializeToString())
  f.close()

