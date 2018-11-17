from numpy.random import seed
seed(1)
import tensorflow.contrib.keras as kr
import tensorflow as tf
import numpy as np
from tensorflow import set_random_seed
set_random_seed(2)
def BNN(xtrain, ytrain, xtest, learning_rate = 0.5, epochs = 100, batch_size = 1000,
       input_size = 6, hidden_size = 5, num_classes =2, kp = 1.0):
    keep_prob = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, [None, input_size], name='x')
    y = tf.placeholder(tf.float32, [None, num_classes], name='y')
    # the weights and biases of input_layer and initializer
    W1 = tf.Variable(tf.random_normal([input_size, hidden_size], stddev = 0.03), name = 'W1')
    b1 = tf.Variable(tf.random_normal([hidden_size]), name = 'b1')

    # the weights and biases of hidden_layer and initializer
    W2 = tf.Variable(tf.random_normal([hidden_size, num_classes], stddev = 0.03), name = 'W2')
    b2 = tf.Variable(tf.random_normal([num_classes]), name = 'b2')

    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.dropout(hidden_out, keep_prob)
    hidden_out = tf.nn.relu(hidden_out)
    
    # softmax function as the output
    y_ = tf.nn.softmax(tf.nn.dropout(tf.add(tf.matmul(hidden_out, W2), b2), keep_prob))

    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    # cross entropy as the loss function
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) 
                                                  + (1 - y) * tf.log(1 - y_clipped), axis=1))
    # optimising by SGD
    optimiser = tf.train.GradientDescentOptimizer(learning_rate =
                                                  learning_rate).minimize(cross_entropy)
    init_op = tf.global_variables_initializer()
    # calculating the correct prediction 
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

    # label argmax 
    T_f = tf.argmax(y_, 1)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #f1_score = tf.contrib.metrics.recall_at_precision(tf.argmax(y,1), tf.argmax(y_, 1))
    
    with tf.Session() as sess:
        sess.run(init_op)
        total_batch = int(len(xtrain)/batch_size)
        for epoch in range(epochs):
            avg_cost = 0 
            for batch_x, batch_y in batch_iter(xtrain, ytrain, batch_size):
                # encoding to one-hot 
                batch_y = kr.utils.to_categorical(batch_y, num_classes)
                _, c = sess.run([optimiser, cross_entropy], 
                               feed_dict = {x: batch_x, y: batch_y, keep_prob: kp} )
                avg_cost += c/total_batch
            #if np.mod(epoch, 20) == 0 :
                #print("Epoch:", (epoch + 1), "cost = ","{:.3f}".format(avg_cost))
        ytest_OH = np.zeros((len(xtest),len(np.unique(ytrain))))
        y_proba, y_pred = sess.run([y_ ,T_f], feed_dict={x: xtest, y: ytest_OH, keep_prob: 1.0})
    return y_pred, y_proba, sess

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
