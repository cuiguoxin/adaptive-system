import tensorflow as tf

def create_sarsa_model():
	with tf.variable_scope("first_layer"):
		placeholder_state = tf.placeholder(tf.float32, [7, 1], name="state")
		variable_first_layer = tf.get_variable("weight", [10, 7], tf.float32,
										initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
		first_layer = tf.matmul(variable_first_layer, placeholder_state);
		activate_first_layer = tf.nn.relu(first_layer)

	with tf.variable_scope("second_layer"):
		variable_second_layer = tf.get_variable("weight", [5, 10], tf.float32, 
										  initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
		second_layer = tf.matmul(variable_second_layer, activate_first_layer)

	second_layer = tf.reshape(second_layer, [5])
	placeholder_one_hot = tf.placeholder(tf.float32, [5], name="one_hot")
	action_value = tf.einsum('i,i->', placeholder_one_hot, second_layer)

	placeholder_learning_rate = tf.placeholder(tf.float32,shape=(),name="learning_rate")
	opt = tf.train.GradientDescentOptimizer(learning_rate=placeholder_learning_rate)
	grads_vars = opt.compute_gradients(action_value)
	training_op = opt.apply_gradients(grads_vars)

	init = tf.global_variables_initializer()
	print init.__name__

	sess = tf.Session()
	tf.train.write_graph(sess.graph_def, './', 'sarsa.pb', as_text=False)


create_sarsa_model()


    
