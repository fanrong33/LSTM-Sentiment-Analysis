# encoding: utf-8
# 评估训练的模型

import numpy as np
import tensorflow as tf


word_vectors = np.load('wordVectors.npy')
print('Loaded the word vectors')

ids = np.load('idsMatrix.npy')


# 辅助函数
from random import randint

def get_train_batch(batch_size, max_seq_length):
    labels = []
    arr = np.zeros([batch_size, max_seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0]) # 正
        else:
            num = randint(13499, 24999)
            labels.append([0, 1]) # 负
        arr[i] = ids[num-1:num]
    return arr, labels

def get_test_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_length])
    for i in range(batch_size):
        num = randint(11499, 13499)
        if num <= 12499:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num-1:num]
    return arr, labels


# 定义神经网络
max_seq_length = 250
batch_size     = 24
lstm_units     = 64
num_classes    = 2
iterations     = 100000 # 100000
num_dimensions = 300 # 每个词向量的维度


labels = tf.placeholder(tf.float32, [batch_size, num_classes])
input_data = tf.placeholder(tf.int32, [batch_size, max_seq_length])

data = tf.Variable(tf.zeros([batch_size, max_seq_length, num_dimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(word_vectors, input_data)


lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)


weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
bias   = tf.Variable(tf.constant(0.1, shape=[num_classes]))

value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)



correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))




saver = tf.train.Saver()

with tf.Session() as sess:
    model_file = tf.train.latest_checkpoint('models') # models/
    if model_file: # 恢复模型
        saver.restore(sess, model_file)
    else:
        print('No checkpoint file found')
        sys.exit()

    iterations = 10
    for i in range(iterations):
        batch_xs, batch_ys = get_test_batch()
        test_accuracy = sess.run(accuracy, feed_dict={input_data:batch_xs, labels: batch_ys})
        print("Accuracy for this batch: %.4f" % test_accuracy)



