# encoding: utf-8
"""
因为需要用到 tf.nn.embedding_lookup, 所以输入的为 input_ids, 而不是最终的 sequence vector
所以 1、训练数据，需要预先整合处理为 [id索引矩阵], 并保存为文件（处理耗时）
    2、待预测的数据，也需要事先整合处理为 [id索引矩阵]，再喂给模型

@version 1.0.0 build 20180303
"""

import os
from os.path import isfile, join
import numpy as np


# 加载词数组列表
words_list = np.load('wordsList.npy')
words_list = words_list.tolist() #Originally loaded as numpy array
words_list = [word.decode('UTF-8') for word in words_list] #Encode words as UTF-8
print('Loaded the word list')


positive_files = ['positiveReviews/'+f for f in os.listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
negative_files = ['negativeReviews/'+f for f in os.listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
        

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


