from preprocess import TextProcessor
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec

class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        label = self.labels[index]
        return text, label

class DataProcessor:
    def __init__(self, word2vec_model, texts, labels_id, split1=0.4, split2=0.5, batch_size=32):
        self.word2vec_model = word2vec_model
        self.texts = texts
        self.labels_id = labels_id
        self.split1 = split1
        self.split2 = split2
        self.batch_size = batch_size
        
        self.text2indexs = self.texts_to_index(self.texts)

        self.train_data, self.val_data, self.train_labels, self.val_labels = self.split_data()
        self.train_loader = self.create_dataloader(self.train_data, self.train_labels)
        self.val_loader = self.create_dataloader(self.val_data, self.val_labels)

    def texts_to_index(self, texts):
        text2indexs = []
        max_length = max(len(text) for text in texts)
        for text in tqdm(texts,desc='文本转索引'):
            text2index = [self.word2vec_model.wv.key_to_index[word] for word in text if word in self.word2vec_model.wv.key_to_index]
            padded_text2index = text2index + [165610] * (max_length - len(text2index)) 
            text2indexs.append(padded_text2index)
        print("文本已转换成索引！")
        return text2indexs

    def split_data(self):
        print("划分数据...")
        train_data, rest_data, train_labels, rest_labels = train_test_split(self.text2indexs, self.labels_id, test_size=self.split1,random_state=42)
        val_data, test_data, val_labels, test_labels = train_test_split(rest_data, rest_labels, test_size=self.split2, random_state=42)
        return train_data, val_data, train_labels, val_labels

    def create_dataloader(self, data, labels):
        print("创建数据加载器...")
        dataset = TextDataset(data, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


'''
text_processor = TextProcessor()
texts = text_processor.texts
label_ids = text_processor.labels_id
id2label = text_processor.id2label

word2vec_model = Word2Vec.load("model/word2vec.model")

data_processor = DataProcessor(word2vec_model, texts, label_ids)
train_dataloader = data_processor.train_loader
val_dataloader = data_processor.val_loader
'''