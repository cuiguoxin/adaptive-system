import tensorflow as tf

def create_sarsa_model():
	with tf.variable_scope("first_layer"):
		placeholder_state = tf.placeholder(tf.float32, [8], name="state")
		print placeholder_state.name
		state = tf.reshape(placeholder_state, [8, 1])
		variable_first_layer = tf.get_variable("weight", [10, 8],
                                        tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
	        bias_first_layer = tf.get_variable("bias", [10, 1], tf.float32, initializer=tf.constant_initializer())
       		first_layer = tf.matmul(variable_first_layer, state) +  bias_first_layer
		activate_first_layer = tf.tanh(first_layer)

	with tf.variable_scope("second_layer"):
		variable_second_layer = tf.get_variable("weight", [5, 10], tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
       		bias_second_layer = tf.get_variable("bias", [5, 1], tf.float32, initializer=tf.constant_initializer())
        	second_layer = tf.matmul(variable_second_layer, activate_first_layer) + bias_second_layer

	second_layer = tf.reshape(second_layer, [5])
	print second_layer.name
	placeholder_one_hot = tf.placeholder(tf.float32, [5], name="one_hot")
	print placeholder_one_hot.name
	action_value = tf.einsum('i,i->', placeholder_one_hot, second_layer)
	print action_value.name

	placeholder_learning_rate = tf.placeholder(tf.float32,shape=(),name="learning_rate")
	print placeholder_learning_rate.name
	opt = tf.train.GradientDescentOptimizer(learning_rate=placeholder_learning_rate)
	grads_vars = opt.compute_gradients(action_value)
	training_op = opt.apply_gradients(grads_vars)
	print training_op.name

	init = tf.global_variables_initializer()
	print init.name

	sess = tf.Session()
	tf.train.write_graph(sess.graph_def, './', 'sarsa.pb', as_text=False)


create_sarsa_model()


    
