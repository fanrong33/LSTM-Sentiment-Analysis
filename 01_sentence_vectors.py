# encoding: utf-8
# version 1.0.0 build 20180302
"""
1、词列表     &  词向量列表 对应
  ['i' ...]   [[1,] ...]

2、输入句子，输出句子对应的词列表索引
i thought the movie was good
[    41    804 201534   1005     15   7446      5  13767      0      0]

3、调用embedding_lookup，将词向量嵌入到句子索引中，输出句子的词向量
tf.nn.embedding_lookup(word_vectors, sentence)
"""

import numpy as np

words_list = np.load('wordsList.npy')
print('Loaded the word list')
# print(words_list)
'''
['0' ',' '.' ..., 'rolonda' 'zsombor' 'unk']
'''
words_list = words_list.tolist() #Originally loaded as numpy array
words_list = [word.decode('UTF-8') for word in words_list] #Encode words as UTF-8

word_vectors = np.load('wordVectors.npy')
print('Loaded the word vectors')

# print(len(words_list))
''' 400000 '''
# print(word_vectors.shape)
''' (400000, 50) '''
# print(word_vectors)
'''
[[ 0.          0.          0.         ...,  0.          0.          0.        ]
 [ 0.013441    0.23682    -0.16899    ..., -0.56656998  0.044691    0.30392   ]
 [ 0.15164     0.30177    -0.16763    ..., -0.35652     0.016413    0.10216   ]
 ...,
 [-0.51181     0.058706    1.09130001 ..., -0.25003001 -1.125       1.58630002]
 [-0.75897998 -0.47426     0.47369999 ...,  0.78953999 -0.014116
   0.64480001]
 [-0.79149002  0.86616999  0.11998    ..., -0.29995999 -0.0063003
   0.39539999]]
'''

baseball_index = words_list.index('baseball')
print(word_vectors[baseball_index])
'''
[-1.93270004  1.04209995 -0.78514999  0.91033     0.22711    -0.62158
 -1.64929998  0.07686    -0.58679998  0.058831    0.35628     0.68915999
 -0.50598001  0.70472997  1.26639998 -0.40031001 -0.020687    0.80862999
 -0.90565997 -0.074054   -0.87674999 -0.62910002 -0.12684999  0.11524
 -0.55685002 -1.68260002 -0.26291001  0.22632     0.713      -1.08280003
  2.12310004  0.49869001  0.066711   -0.48225999 -0.17896999  0.47699001
  0.16384     0.16537    -0.11506    -0.15962    -0.94926    -0.42833
 -0.59456998  1.35660005 -0.27506     0.19918001 -0.36008     0.55667001
 -0.70314997  0.17157   ]
'''

import tensorflow as tf

max_seq_length = 10  # 句子的长度
num_dimensions = 300 # 每个单词向量的维度
first_sentence = np.zeros((max_seq_length), dtype='int32')
first_sentence[0] = words_list.index('i')
first_sentence[1] = words_list.index('thought')
first_sentence[2] = words_list.index('the')
first_sentence[3] = words_list.index('movie')
first_sentence[4] = words_list.index('was')
first_sentence[5] = words_list.index('incredible')
first_sentence[6] = words_list.index('and')
first_sentence[7] = words_list.index('inspiring')
print(first_sentence.shape)
''' (10,) '''

print(first_sentence) # 索引位置
''' [    41    804 201534   1005     15   7446      5  13767      0      0] '''

with tf.Session() as sess:
    # 查表嵌入
    print(tf.nn.embedding_lookup(word_vectors, first_sentence).eval().shape)
    '''
    (10, 50)
    '''


