import jieba
from tqdm import tqdm
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import torch

#数据预处理：从数据集中提取文本和标签
class TextProcessor:
    def __init__(self, dataset_path= './dataset/dataset.txt', stopwords_path = './dataset/cn_stopwords.txt'):
        self.dataset_path = dataset_path
        self.stopwords = self.get_stopwords(stopwords_path)
        self.texts, self.labels = self.co_data()
        self.id2label, self.label2id = self.co_labeldict()
        self.vocab=self.co_word2idx(self.texts)
        self.labels_id = [self.label2id[label] for label in self.labels]

    #获取停用词
    def get_stopwords(self, stop_file_name):
        with open(stop_file_name, "r", encoding="utf-8") as file:
            lines = file.readlines()
        words = [i.strip() for i in lines]
        return words

    #提取文本
    def text_cleaning(self, text):
        text_result = ''
        for char in text:
            if '\u4e00' <= char <= '\u9fa5':
                text_result += char
        return text_result

    #提取分词后的文本和标签
    def co_data(self):
        labels = []
        texts = []

        with open(self.dataset_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                parts = line.split("_!_")
                labels.append(parts[2])
                texts.append(parts[3])

        # 对文本分词并去除停用词
        temp = texts.copy()
        texts = []
        for text in tqdm(temp,desc='分词并去除停用词'):
            result = self.text_cleaning(text)
            seg = jieba.cut(result, cut_all=False)
            text = [char for char in seg if char not in self.stopwords]
            texts.append(text)
        return texts, labels
    
    #构建标签对应的id
    def co_labeldict(self):
        label_freq = dict(Counter(self.labels))

        id2labels = {i: label for i, label in enumerate(label_freq)}
        labels2id = {label: i for i, label in enumerate(label_freq)}
        return id2labels, labels2id
    
    # 构建文字-索引字典
    def co_word2idx(self,texts):
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
                data_vec = data_vec + [len(word2idx)+1] * padding_len
            dataset_list.append(data_vec)
        return dataset_list

#将文本划分为train,val和test并加载loader
class DataProcessor:
    def __init__(self, texts, labels_id, vocab, split1=0.4, split2=0.5, batch_size=32):
        self.texts = texts
        self.labels_id = labels_id
        self.vocab = vocab
        self.split1 = split1
        self.split2 = split2
        self.batch_size = batch_size

        self.train_data, self.val_data, self.train_labels, self.val_labels, self.test_data, self.test_labels= self.split_data()
        self.train_data = self.co_vec(self.train_data, self.vocab)
        self.val_data = self.co_vec(self.val_data, self.vocab)
        self.test_data = self.co_vec(self.test_data, self.vocab)

        self.train_loader = self.create_dataloader(self.train_data, self.train_labels)
        self.val_loader = self.create_dataloader(self.val_data, self.val_labels)
        self.test_loader = self.create_dataloader(self.test_data, self.test_labels)

    #将三个数据集的中的文字转化为索引，并且将每段文字裁剪或扩展为100个字
    def co_vec(self, dataset_data, word2idx):
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
                data_vec = data_vec + [len(word2idx)+1] * padding_len
            dataset_list.append(data_vec)
        return dataset_list

    def split_data(self):
        '''
        划分数据集
        训练集：60%
        验证集：20%
        测试集：20%
        '''
        print("划分数据...")
        train_data, rest_data, train_labels, rest_labels = train_test_split(self.texts, self.labels_id, test_size=self.split1,random_state=42)
        val_data, test_data, val_labels, test_labels = train_test_split(rest_data, rest_labels, test_size=self.split2, random_state=42)
        return train_data, val_data, train_labels, val_labels,test_data,test_labels

    def create_dataloader(self, data, labels):
        print("创建数据加载器...")
        dataset = TextDataset(data, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

# 创建自定义的数据集类
class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data)
        self.labels=torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        label = self.labels[index]
        return text, label

'''
text_processor = TextProcessor()
texts = text_processor.texts
vocab = text_processor.vocab
label_ids = text_processor.labels_id
id2label = text_processor.id2label
label2id = text_processor.label2id

data_processor = DataProcessor(texts, label_ids, vocab)
train_dataloader = data_processor.train_loader
val_dataloader = data_processor.val_loader
test_dataloader = data_processor.test_loader
'''
