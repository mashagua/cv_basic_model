import tensorflow as tf
import numpy as np
x=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x.shape)
y=np.square(x)-0.5+noise
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])
def add_layer(inputs,in_size,out_size,activation_function=None):
    weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    wx_plus_b=tf.matmul(inputs,weights)+biases
    if activation_function is None:
        outputs=wx_plus_b
    else:
        outputs=activation_function(wx_plus_b)
    return outputs
#到隐含层10个
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
#到输出层1个
prediction = add_layer(l1, 10, 1, activation_function=None)
#
loss=tf.reduce_mean(tf.reduce_mean(ys-prediction),reduction_indices=[1])