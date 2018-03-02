# encoding: utf-8
# 预测

import numpy as np
import tensorflow as tf

words_list = np.load('wordsList.npy')
words_list = words_list.tolist() #Originally loaded as numpy array
words_list = [word.decode('UTF-8') for word in words_list] #Encode words as UTF-8
print('Loaded the word list')

word_vectors = np.load('wordVectors.npy')
print('Loaded the word vectors')


# RNN Model
max_seq_length = 250
lstm_units     = 64
num_classes    = 2
num_dimensions = 300 # 每个词向量的维度
iterations     = 100000 # 100000
batch_size     = 24



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

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, '', string.lower())

def get_sentence_matrix(sentence):
    sentence_matrix = np.zeros([batch_size, max_seq_length], dtype='int32')
    cleaned_sentence = clean_sentences(sentence)
    split = cleaned_sentence.split()
    index = 0
    for word in split:
        try:
            sentence_matrix[0, index] = words_list.index(word)
        except ValueError:
            sentence_matrix[0, index] = 399999
        index = index + 1
        if index >= max_seq_length:
            break
    return sentence_matrix



saver = tf.train.Saver()

with tf.Session() as sess:
    model_file = tf.train.latest_checkpoint('models') # models/
    if model_file: # 恢复模型
        saver.restore(sess, model_file)
    else:
        print('No checkpoint file found')
        sys.exit()


    input_text_1 = "That movie was terrible."
    input_matrix = get_sentence_matrix(input_text_1)

    predict = sess.run(prediction, feed_dict={input_data: input_matrix})
    predict = predict[0]
    print(predict)
    # predict[0] represents output score for positive sentiment
    # predict[1] represents output score for negative sentiment
    if (predict[0] > predict[1]):
        print "Positive Sentiment"
    else:
        print "Negative Sentiment"

    input_text_2 = "That movie was the best one I have ever seen."
    input_matrix = get_sentence_matrix(input_text_2)

    predict = sess.run(prediction, feed_dict={input_data: input_matrix})
    predict = predict[0]
    print(predict)
    # predict[0] represents output score for positive sentiment
    # predict[1] represents output score for negative sentiment
    if (predict[0] > predict[1]):
        print "Positive Sentiment"
    else:
        print "Negative Sentiment"


        