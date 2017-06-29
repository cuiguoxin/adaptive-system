import tensorflow as tf

def create_actor_critic_model():
	with tf.variable_scope("value_function/first_layer"):
		placeholder_state = tf.placeholder(tf.float32, [8], name="state")
		print placeholder_state.name
		state = tf.reshape(placeholder_state, [8, 1])
		variable_first_layer = tf.get_variable("weight", [10, 8],
                                        tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
	        bias_first_layer = tf.get_variable("bias", [10, 1], tf.float32, initializer=tf.constant_initializer())
       		first_layer = tf.matmul(variable_first_layer, state) +  bias_first_layer
		activate_first_layer = tf.tanh(first_layer)

	with tf.variable_scope("value_function/second_layer"):
		variable_second_layer = tf.get_variable("weight", [1, 10], tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
       		bias_second_layer = tf.get_variable("bias", [1, 1], tf.float32, initializer=tf.constant_initializer())
        	second_layer = tf.matmul(variable_second_layer, activate_first_layer) + bias_second_layer

    second_layer = tf.reshape(second_layer, [1])
	print second_layer.name
    placeholder_learning_rate = tf.placeholder(tf.float32,shape=(),name="value_function/learning_rate")
	print placeholder_learning_rate.name
	opt = tf.train.GradientDescentOptimizer(learning_rate=placeholder_learning_rate)
	grads_vars = opt.compute_gradients(second_layer)
	training_op = opt.apply_gradients(grads_vars)
	print training_op.name

    with tf.variable_scope("policy/first_layer"):
		placeholder_state = tf.placeholder(tf.float32, [8], name="state")
		print placeholder_state.name
		state = tf.reshape(placeholder_state, [8, 1])
		variable_first_layer = tf.get_variable("weight", [10, 8],
                                        tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
	        bias_first_layer = tf.get_variable("bias", [10, 1], tf.float32, initializer=tf.constant_initializer())
       		first_layer = tf.matmul(variable_first_layer, state) +  bias_first_layer
		activate_first_layer = tf.tanh(first_layer)

	with tf.variable_scope("policy/second_layer"):
		variable_second_layer = tf.get_variable("weight", [3, 10], tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
       		bias_second_layer = tf.get_variable("bias", [3, 1], tf.float32, initializer=tf.constant_initializer())
        	policy_second_layer = tf.matmul(variable_second_layer, activate_first_layer) + bias_second_layer

	policy_second_layer = tf.reshape(policy_second_layer, [3])
    policy_second_layer = tf.nn.log_softmax(policy_second_layer)

	print policy_second_layer.name
	placeholder_one_hot = tf.placeholder(tf.float32, [3], name="one_hot")
	print placeholder_one_hot.name
	policy_value = tf.einsum('i,i->', placeholder_one_hot, policy_second_layer)
	

	policy_placeholder_learning_rate = tf.placeholder(tf.float32,shape=(),name="policy/learning_rate")
	print placeholder_learning_rate.name
	opt = tf.train.GradientDescentOptimizer(learning_rate=policy_placeholder_learning_rate)
	grads_vars = opt.compute_gradients(policy_value)
	training_op = opt.apply_gradients(grads_vars)
	print training_op.name

	init = tf.global_variables_initializer()
	print init.name

	sess = tf.Session()
	tf.train.write_graph(sess.graph_def, './', 'actor_critic_continuous.pb', as_text=False)


create_actor_critic_model()


    
