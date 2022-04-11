import tensorflow.compat.v1 as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.disable_v2_behavior()

A = [1,3,5,7,9]
B = [2,4,6,8,10]

ph_A = tf.placeholder(dtype = tf.float32)
ph_B = tf.placeholder(dtype = tf.float32)

result_sum = ph_A + ph_B

sess = tf.Session()
result = sess.run(result_sum, feed_dict ={ph_A:A, ph_B:B})

print(result)