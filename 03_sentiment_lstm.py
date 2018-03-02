# encoding: utf-8
# version 1.0.0 build 20180302

import numpy as np

# 加载词向量
words_list = np.load('wordsList.npy')
print('Loaded the word list')

words_list = words_list.tolist() #Originally loaded as numpy array
words_list = [word.decode('UTF-8') for word in words_list] #Encode words as UTF-8

word_vectors = np.load('wordVectors.npy')
print('Loaded the word vectors')



import os
import codecs

positive_files = []
negative_files = []
for f in os.listdir('positiveReviews/'):
    if os.path.isfile(os.path.join('positiveReviews/', f)):
        positive_files.append('positiveReviews/'+f)
for f in os.listdir('negativeReviews/'):
    if os.path.isfile(os.path.join('negativeReviews/', f)):
        negative_files.append('negativeReviews/'+f)


# 让我们看看我们如何获取单个文件并将其转换为我们的ID矩阵。
fname = positive_files[3]


max_seq_length = 250

import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "",string.lower())

# 评论句子对应的词向量索引矩阵
first_file = np.zeros((max_seq_length), dtype='int32')
with open(fname) as f:
    index = 0
    line = f.readline()
    cleaned_line = clean_sentences(line)
    split = cleaned_line.split()
    for word in split:
        if index < max_seq_length:
            try:
                first_file[index] = words_list.index(word)
            except ValueError:
                first_file[index] = 399999
        index = index + 1

print(first_file)      
'''
[    37     14   2407 201534     96  37314    319   7158 201534   6469
   8828   1085     47   9703     20    260     36    455      7   7284
   1139      3  26494   2633    203    197   3941  12739    646      7
   7284   1139      3  11990   7792     46  12608    646      7   7284
   1139      3   8593     81  36381    109      3 201534   8735    807
   2983     34    149     37    319     14    191  31906      6      7
    179    109  15402     32     36      5      4   2933     12    138
      6      7    523     59     77      3 201534     96   4246  30006
    235      3    908     14   4702   4571     47     36 201534   6429
    691     34     47     36  35404    900    192     91   4499     14
     12   6469    189     33   1784   1318   1726      6 201534    410
     41    835  10464     19      7    369      5   1541     36    100
    181     19      7    410      0      0      0      0      0      0
      0      0      0      0      0      0      0      0      0      0
      0      0      0      0      0      0      0      0      0      0
      0      0      0      0      0      0      0      0      0      0
      0      0      0      0      0      0      0      0      0      0
      0      0      0      0      0      0      0      0      0      0
      0      0      0      0      0      0      0      0      0      0
      0      0      0      0      0      0      0      0      0      0
      0      0      0      0      0      0      0      0      0      0
      0      0      0      0      0      0      0      0      0      0
      0      0      0      0      0      0      0      0      0      0
      0      0      0      0      0      0      0      0      0      0
      0      0      0      0      0      0      0      0      0      0]
'''


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


# RNN Model
batch_size     = 24
lstm_units     = 64
num_classes    = 2
iterations     = 100000 # 100000
num_dimensions = 300 # 每个词向量的维度

# 辅助函数
from random import randint

def get_train_batch():
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


import tensorflow as tf

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



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)



import datetime

tf.summary.scalar('Loss', cost)
tf.summary.scalar('Accuracy', accuracy)

saver = tf.train.Saver()

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    merged = tf.summary.merge_all()
    logdir = 'tensorboard/'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'/'
    writer = tf.summary.FileWriter(logdir, sess.graph)


    for i in range(iterations):
        batch_xs, batch_ys = get_train_batch()
        sess.run(optimizer, feed_dict={input_data: batch_xs, labels: batch_ys})

        # Write summary to Tensorboard
        if i % 50 == 0:
            loss = sess.run(cost, feed_dict={input_data:batch_xs, labels: batch_ys})
            train_accuracy = sess.run(accuracy, feed_dict={input_data:batch_xs, labels: batch_ys})
            print('Step %s, Training Accuracy: %.4f, Minibatch Loss: %.3f' % (i, train_accuracy, loss))

            summary = sess.run(merged, feed_dict={input_data: batch_xs, labels: batch_ys})
            writer.add_summary(summary, i)

        # Save the network every 10,000 training iterations
        if i % 10000 == 0:
            save_path = saver.save(sess, 'models/pretrained_lstm2.ckpt', global_step=i)
            print('saved to %s' % save_path)

    writer.close()

    print("Optimization Finished!")






