import tensorflow as tf
import sys

hidden_layer_size = 30


def create_sarsa_model(input_size, output_size):
    with tf.variable_scope("first_layer"):
        placeholder_state = tf.placeholder(
            tf.float32, [input_size], name="state")
        print placeholder_state.name
        state = tf.reshape(placeholder_state, [input_size, 1])
        variable_first_layer = tf.get_variable(
            "weight", [hidden_layer_size, input_size],
            tf.float32,
            initializer=tf.truncated_normal_initializer(
                stddev=5e-2, dtype=tf.float32))
        bias_first_layer = tf.get_variable(
            "bias", [hidden_layer_size, 1],
            tf.float32,
            initializer=tf.constant_initializer())
        first_layer = tf.matmul(variable_first_layer, state) + bias_first_layer
        activate_first_layer = tf.tanh(first_layer)

    with tf.variable_scope("second_layer"):
        variable_second_layer = tf.get_variable(
            "weight", [output_size, hidden_layer_size],
            tf.float32,
            initializer=tf.truncated_normal_initializer(
                stddev=5e-5, dtype=tf.float32))
        bias_second_layer = tf.get_variable(
            "bias", [output_size, 1],
            tf.float32,
            initializer=tf.constant_initializer())
        second_layer = tf.matmul(variable_second_layer,
                                 activate_first_layer) + bias_second_layer

    second_layer = tf.reshape(second_layer, [output_size])
    print second_layer.name
    placeholder_one_hot = tf.placeholder(
        tf.float32, [output_size], name="one_hot")
    print placeholder_one_hot.name
    action_value = tf.einsum('i,i->', placeholder_one_hot, second_layer)
    print action_value.name

    placeholder_learning_rate = tf.placeholder(
        tf.float32, shape=(), name="learning_rate")
    print placeholder_learning_rate.name
    opt = tf.train.GradientDescentOptimizer(
        learning_rate=placeholder_learning_rate)
    grads_vars = opt.compute_gradients(action_value)
    training_op = opt.apply_gradients(grads_vars)
    print training_op.name

    init = tf.global_variables_initializer()
    print init.name

    sess = tf.Session()
    tf.train.write_graph(
        sess.graph_def, './', 'sarsa_continous.pb', as_text=False)


input_size = int(sys.argv[1])
output_size = int(sys.argv[2])
create_sarsa_model(input_size, output_size)
