import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    a = tf.Variable(5.0, name='a')
    b = tf.Variable(6.0, name='b')
    c = tf.multiply(a, b, name="c")

    opt = GradientDescentOptimizer(learning_rate=0.1)
    grads_and_vars = opt.compute_gradients(c, [a, b])
   
    init = tf.global_variables_initializer()
    print init.name
    print a.name
    sess.run(init)

    print a.eval() # 5.0
    print b.eval() # 6.0
    print c.eval() # 30.0

    for grad_and_var in grads_and_vars:
        print grad_and_var[0].name
        print grad_and_var[0].eval()

    
    tf.train.write_graph(sess.graph_def, '../load_graph/graph_pb/', 'graph.pb', as_text=False)
