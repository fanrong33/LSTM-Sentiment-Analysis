# encoding: utf-8
""" 
我们要使用的训练集是imdb电影评论数据集。
这个集合有25,000个电影评论，有12,500个正面评论和12,500个负面评论。
"""

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

# 统计每个评论的总词汇量和平均次数
num_words = [] 
for pf in positive_files:
    with codecs.open(pf, 'r', encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('Possitive files finished')

for nf in negative_files:
    with codecs.open(nf, 'r', encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('Negative files finished')

num_files = len(num_words)
print('The total number of files is %d' % num_files) 
print('The total number of words in the files is %d' % sum(num_words))
print('The average number of words in the files is %d' % (sum(num_words)/num_files))



# 以直方图格式可视化这些数据
import matplotlib.pyplot as plt

plt.hist(num_words, 50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.axis([0, 1200, 0, 8000])
plt.show()
# 根据直方图以及每个文件的平均字数，我们可以安全地说，大多数评论将落在250字以下，这是我们将设置的最大序列长度值



'''
Possitive files finished
Negative files finished
The total number of files is 25000
The total number of words in the files is 5758410
The average number of words in the files is 230
'''
