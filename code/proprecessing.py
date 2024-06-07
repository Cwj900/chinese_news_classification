import jieba
from tqdm import tqdm
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import torch


#获取停用词
def get_stopwords(stop_file_name):
    with open(stop_file_name, "r", encoding="utf-8") as file:
        lines=file.readlines()
    words=[i.strip() for i in lines]
    return words

#字符清洗：
def text_cleaning(text):
    text_result=''
    for char in text:
        if (char>='\u4e00' and char<='\u9fa5') :
            text_result+=char
    return text_result

#数据预处理
def co_data(dataset_path,stopwords):
    labels=[]
    labels_idx=[]
    texts=[]

    with open(dataset_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split("_!_")
            labels_idx.append(parts[1])
            labels.append(parts[2])
            texts.append(parts[3])

    #字符清洗
    temp=texts.copy()
    texts=[]
    for text in tqdm(temp):
        result=text_cleaning(text)
        seg=jieba.cut(result, cut_all=False)
        text=[char for char in seg if not char in stopwords]
        texts.append(text)
    
    return texts,labels,labels_idx

def co_word2idx(texts):
    # 统计所有文字
    word_counts = Counter()
    for text_line in texts:
        word_counts.update(text_line)
    
    # 按照出现次数降序排列
    sorted_words = word_counts.most_common()

    word2idx = {'<UNK>': 0} 
    for idx, (word, count) in enumerate(sorted_words):
        word2idx[word] = idx + 1  # 从1开始编号
    
    word2idx = {key: word2idx[key] for key in list(word2idx.keys())[:7000]}

    return word2idx

#整理类别和索引
def co_labeldict(labels,labels_idx):
    id2label={}
    label2id={}
    for category, value in zip(labels, labels_idx):
        id2label[int(value)] = category
        label2id[category] = int(value)
    return id2label,label2id

#将三个数据集的中的文字转化为索引，并且将每段文字裁剪或扩展为100个字
def co_vec(dataset_data,word2idx):
    dataset_list=[]
    for data in dataset_data:
        data_vec=[]

        for word in data:
            if word in word2idx:
                word_vector=word2idx[word]
            else:
                word_vector=word2idx['<UNK>']
            data_vec.append(word_vector)
        
        #截断
        if len(data_vec)>=100:
            data_vec=data_vec[:100]
        #填充
        else:
            padding_len=100-len(data_vec)
            data_vec = data_vec + [-1] * padding_len
        dataset_list.append(data_vec)
    return dataset_list

# 创建自定义的数据集类
class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data)
        labels = [label2id[label] for label in labels]
        self.labels=torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        label = self.labels[index]
        return text, label


dataset_path='./dataset/dataset.txt'
stopwords=get_stopwords('./dataset/cn_stopwords.txt')
#数据预处理
texts,labels,labels_idx=co_data(dataset_path,stopwords)
word2idx=co_word2idx(texts)
print(len(word2idx))
id2label,label2id=co_labeldict(labels,labels_idx)

# 加载词嵌入模型
# model = KeyedVectors.load_word2vec_format('./word2vec/sgns.sogou.word', binary=False)

'''
划分数据集
训练集：60%
验证集：20%
测试集：20%
'''
#将数据集转换为向量
train_data, rest_data, train_labels, rest_labels = train_test_split(texts, labels, test_size=0.4, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(rest_data, rest_labels, test_size=0.5, random_state=42)
train_vec=co_vec(train_data,word2idx)
val_vec=co_vec(val_data,word2idx)
test_vec=co_vec(test_data,word2idx)

train_dataset=TextDataset(train_vec,train_labels)
val_dataset=TextDataset(val_vec,val_labels)
test_dataset=TextDataset(test_vec,test_labels)

train_dataloader=DataLoader(train_dataset,32,shuffle=True)
val_dataloader=DataLoader(val_dataset,32,shuffle=False)
print(train_vec[0])


