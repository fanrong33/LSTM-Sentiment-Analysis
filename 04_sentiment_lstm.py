# encoding: utf-8
""" 
@version 1.0.0 build 20180302
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np



# ====================
#  Imdb movie review dataset
# ====================
# class MovieSequenceData(object):



# 加载词向量
words_list = np.load('wordsList.npy')
words_list = words_list.tolist() #Originally loaded as numpy array
words_list = [word.decode('UTF-8') for word in words_list] #Encode words as UTF-8
print('Loaded the word list')


word_vectors = np.load('wordVectors.npy')
print('Loaded the word vectors')




# 现在，让我们为我们的25,000条评论中的每条评论做同样的事情。
# 我们将加载电影训练集并将其整合以获得25000 x 250的矩阵。
# 这是一个计算成本很高的过程，所以不是让你运行整块，而是加载一个预先计算好的id矩阵。

# ids = np.zeros((25000, max_seq_length), dtype='int32')
# file_counter = 0
# for pf in positive_files:
#     with open(pf, 'r') as f:
#         index = 0
#         line = f.readline()
#         cleaned_line = clean_sentences(line)
#         split = cleaned_line.split()
#         for word in split:
#             try:
#                 ids[file_counter][index] = words_list.index(word)
#             except ValueError:
#                 ids[file_counter][index] = 399999
#             index = index + 1
#             if index >= max_seq_length:
#                 break
#         file_counter = file_counter + 1

# for nf in negative_files:
#     with open(nf, 'r') as f:
#         index = 0
#         line = f.readline()
#         cleaned_line = clean_sentences(line)
#         split = cleaned_line.split()
#         for word in split:
#             try:
#                 ids[file_counter][index] = words_list.index(word)
#             except ValueError:
#                 ids[file_counter][index] = 399999
#             index = index + 1
#             if index >= max_seq_length:
#                 break
#         file_counter = file_counter + 1
# #Pass into embedding function and see if it evaluates.

# np.save('ids_matrix', ids)


ids = np.load('idsMatrix.npy')


# 辅助函数
from random import randint

def get_train_batch(batch_size, max_seq_length):
    labels = []
    arr = np.zeros([batch_size, max_seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0]) # 正 One-hot编码，跟mnist一样
        else:
            num = randint(13499, 24999)
            labels.append([0, 1]) # 负
        arr[i] = ids[num-1:num]
    return arr, labels

def get_test_batch(batch_size, max_seq_length):
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


# 定义训练参数
learning_rate  = 0.01
training_steps = 100000
batch_size     = 24
display_step   = 50

# 定义神经网络参数
max_seq_length = 250 # Sequence max length
num_hidden  = 64 # hidden layer num of features
num_classes    = 2

lstm_units     = 64
iterations     = 100000 # 100000
num_dimensions = 300 # 每个词向量的维度



# tf Graph input
# 这里x为input_ids用于存储索引
x  = tf.placeholder(tf.int32, shape=[None, max_seq_length])
y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

# Store layers weight 权重 & bias 偏值
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([lstm_units, num_classes], stddev=0.1))
}
biases = {
    # 'out': tf.Variable(tf.random_normal([num_classes]))
    'out': tf.Variable(tf.constant(0.1, shape=[num_classes])),
}



# Create model
def dynamicRNN(x, seqlen, weights, biases):
    
    data = tf.Variable(tf.zeros([batch_size, max_seq_length, num_dimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(word_vectors, x)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)


    value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)
    value = tf.transpose(value, [1, 0, 2])

    outputs = tf.gather(value, int(value.get_shape()[0]) - 1)

    return tf.matmul(outputs, weights['out']) + biases['out']


# Construct model
prediction = dynamicRNN(x, max_seq_length, weights, biases)


# 定义成本函数, 使用tf内置定义的交叉熵函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_))
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(cost)


# Evaluate model 评估模型
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


import datetime

tf.summary.scalar('Loss', cost)
tf.summary.scalar('Accuracy', accuracy)

saver = tf.train.Saver()
# 启动图
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    merged = tf.summary.merge_all()
    logdir = 'tensorboard/'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'/'
    writer = tf.summary.FileWriter(logdir, sess.graph)


    for i in range(iterations):
        batch_xs, batch_ys = get_train_batch(batch_size, max_seq_length)
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

        # Write summary to Tensorboard
        if i % 50 == 0:
            loss = sess.run(cost, feed_dict={x:batch_xs, y_: batch_ys})
            train_accuracy = sess.run(accuracy, feed_dict={x:batch_xs, y_: batch_ys})
            print('Step %s, Training Accuracy: %.4f, Minibatch Loss: %.3f' % (i, train_accuracy, loss))

            summary = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys})
            writer.add_summary(summary, i)

        # Save the network every 10,000 training iterations
        # if i % 10000 == 0:
        #     save_path = saver.save(sess, 'models/pretrained_lstm2.ckpt', global_step=i)
        #     print('saved to %s' % save_path)

    writer.close()

    print("Optimization Finished!")

# 定位到本地目录并在终端运行以下命令
# $ tensorboard --logdir=tensorboard





