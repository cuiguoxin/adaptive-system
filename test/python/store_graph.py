import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    a = tf.Variable(5.0, name='a')
    b = tf.Variable(6.0, name='b')
    c = tf.multiply(a, b, name="c")

    init = tf.global_variables_initializer()
    print init.name
    print a.name
    sess.run(init)

    print a.eval() # 5.0
    print b.eval() # 6.0
    print c.eval() # 30.0
    
    tf.train.write_graph(sess.graph_def, '../load_graph/graph_pb/', 'graph.pb', as_text=False)
