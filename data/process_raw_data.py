"""
将正负评论的aspect提出
组成
[   test,
    [ aspect: sentiment]
]
形式
"""
import random

import pandas as pd
import numpy as np


import json


def get_label(label):
    res = list()
    for key, value in label.items():
        if key == '位置':
            continue
        elif key == '情感':
            if value == '好':
                return True
            else:
                return False
        else:
            now_label = get_label(value)
            if now_label == True or now_label == False:
                res += [(key, now_label)]
            elif now_label:
                res += now_label
    return res


file = "sentiments522.txt"
data_list = list()
with open(file, 'r', encoding='utf-8') as fp:
    lines = fp.readlines()
    for line in lines:
        load_dict = json.loads(line)
        data_list.append(load_dict)
print(data_list[0])

test_flag_dict = dict()

data_text = list()
data_label = list()
for i in range(len(data_list)):
    now_text = data_list[i]['text']
    if now_text not in test_flag_dict.keys():
        now_p = random.random()
        if now_p <= 0.1:
            test_flag_dict[now_text] = True
        else:
            test_flag_dict[now_text] = False
    label = data_list[i]['label']
    now_label = get_label(label)
    for j in range(len(now_label)):
        data_text.append(now_text)
        data_label.append(now_label[j])

print(len(data_text))

file = "bad_comments.csv"
with open(file, 'r', encoding='utf-8') as fp:
    lines = fp.readlines()
    for line in lines[1:]:
        line = line.strip().split(',')
        now_text = line[0]
        if now_text not in test_flag_dict.keys():
            now_p = random.random()
            if now_p <= 0.1:
                test_flag_dict[now_text] = True
            else:
                test_flag_dict[now_text] = False
        now_label = line[1:]
        for label in now_label:
            if label != "":
                label = label.split('-')
                print(label)
                if label[-1] == '好':
                    data_text.append(now_text)
                    data_label.append((label[-2], True))
                else:
                    data_text.append(now_text)
                    data_label.append((label[-2], False))

data_aspect = [_[0] for _ in data_label]
data_label = [_[1] for _ in data_label]
all_data = list(zip(data_text, data_aspect, data_label))
data_df = pd.DataFrame(all_data, columns=['text', 'aspect', 'label'])
data_df.to_csv('all_data.csv')

random.seed(1)
random.shuffle(all_data)
train_data = list()
test_data = list()
for i in range(len(all_data)):
    if test_flag_dict[all_data[i][0]]:
        test_data.append(all_data[i])
    else:
        train_data.append(all_data[i])
# train_size = int(len(all_data)*0.9)
# train_data = all_data[:train_size]
# test_data = all_data[train_size:]
train_data_df = pd.DataFrame(train_data, columns=['text', 'aspect', 'label'])
train_data_df.to_csv('train_data.csv')
test_data_df = pd.DataFrame(test_data, columns=['text', 'aspect', 'label'])
test_data_df.to_csv('test_data.csv')
