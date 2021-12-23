# PL-ABSA
Prompt Learning - Aspect-based Sentiment Analysis

使用OpenPrompt进行Prompt Learning在ABSA(aspect-based sentiment analysis)上的实验。

## 环境
使用config.txt安装所需环境

## 代码
### 训练
使用train.py进行基于Prompt Learning的ABSA训练，训练好的模型会被保存为pl-absa.pt

使用bert_train.py进行基于Bert拼句的ABSA训练

### 推理
使用inference.py载入训练好的Prompt Learning模型进行测试集的推理，预测的结果会被输出为predictions.csv

## 数据
### 训练&验证输入
使用形如data/train.csv的文件进行训练数据的输入,
即header为
,text,aspect,label
的csv文件

### 测试输入
使用形如data/test.csv的文件进行测试数据的输入,
即header为
,text,aspect
的csv文件

### 测试输出
推理生成的文件是header为
,text,aspect,prediction
的csv文件
