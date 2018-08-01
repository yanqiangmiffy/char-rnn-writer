from collections import Counter
import os
import sys
import numpy as np

# 开始和结束标志
start_token='B'
end_token='E'

def build_dataset(filename):
    """
    构建数据集 词典以及诗向量
    :param filename: 诗集文件
    :return:
    """
    poems=[]
    no_char = '_(（《['
    with open(filename,'r',encoding='utf-8') as in_data:
        for poem in in_data.readlines():
            try:
                title,content=poem.strip().split(':') # 标题与内容
                content=content.replace(' ','') # 去除空格

                if set(no_char) & set(content): # 将含有特殊标点符号的去掉
                    continue

                if len(content)<5 or len(content)>79: # 将不符合规定长度的去掉
                    continue
                content=start_token+content+end_token
                poems.append(content)
            except ValueError as e:
                pass
    # print(poems)
    words_list=[word for poem in poems for word in poem] # 两层嵌套，插眼
    # print(words_list)
    # print(len(set(words_list)))
    counter=Counter(words_list).most_common()
    # print(counter)
    words,_=zip(*counter)
    words = words + (' ',)
    # print(words)
    word_to_int=dict(zip(words,range(len(words))))
    # print(word_to_int)
    # int_to_word=dict(zip(word_to_int.values(),word_to_int.keys()))
    # print(int_to_word)

    poems_vector = [list(map(lambda word: word_to_int.get(word, len(words)), poem)) for poem in poems] # 默认值为空 len(words) 46 ' '

    # print(poems_vector)
    return poems_vector,word_to_int,words

# poems_vector,word_to_int,words=build_dataset('data/demo.txt')

def generate_batch(batch_size,poems_vector,word_to_int):
    num_batch=len(poems_vector)//batch_size
    x_batches=[]
    y_batches=[]

    for i in range(num_batch):
        start_index=i*batch_size
        end_index=(i+1)* batch_size

        batches=poems_vector[start_index:end_index]
        max_length=max(map(len,batches))
        x_data=np.full((batch_size,max_length),word_to_int[' '],np.int32)
        for row,batch in enumerate(batches):
            x_data[row,:len(batch)]=batch
        y_data=np.copy(x_data)
        y_data[:,:-1]=y_data[:,1:]

        """
        x:3 12 13 14 15 16
        y:12 13 14 15 16 0
        """
        x_batches.append(x_data)
        y_batches.append(y_data)

    return x_batches,y_batches

# generate_batch(1,poems_vector,word_to_int)