from preprocess import TextProcessor
from text2vector import DataProcessor

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from gensim.models import Word2Vec

class CNN_LSTM(nn.Module):
    def __init__(self,num_classes,vocab_size,embedding_dim,embedding_matrix) :
        super(CNN_LSTM,self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_layer.weight.data.copy_(embedding_matrix)
        self.embedding_layer.weight.requires_grad = True  
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(64, 100, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(100, num_classes)


    def forward(self, inputs):
        embedded = self.embedding_layer(inputs)
        embedded = embedded.permute(0, 2, 1)
        out = self.conv1(embedded)
        out = torch.relu(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = torch.relu(out)
        out = self.maxpool2(out)
        out, _ = self.lstm(out.transpose(1, 2))
        out = out[:, -1, :] 
        output = self.fc(out)
        return output

#数据预处理
text_processor = TextProcessor()
texts = text_processor.texts
label_ids = text_processor.labels_id
id2label = text_processor.id2label
#加载Word2Vec模型
word2vec_model = Word2Vec.load("model/word2vec.model")
#划分数据
data_processor = DataProcessor(word2vec_model, texts, label_ids)
train_dataloader = data_processor.train_loader
val_dataloader = data_processor.val_loader

#初始化嵌入层
vocab_size = len(word2vec_model.wv.key_to_index) + 1  
embedding_dim=100
# 创建嵌入矩阵
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, index in word2vec_model.wv.key_to_index.items():
    embedding_matrix[index] = word2vec_model.wv[word]
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_matrix=embedding_matrix.to(device)
model = CNN_LSTM (15,vocab_size,embedding_dim,embedding_matrix).to(device) 
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) 

# 训练模型
max_epochs = 100

for epoch in range(max_epochs):
    model.train()  # 设置模型为训练模式
    for inputs, labels in tqdm(train_dataloader):
        inputs,labels=inputs.long().to(device),labels.long().to(device)
        optimizer.zero_grad()  # 梯度清零

        # 前向传播
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()
    scheduler.step()  # 更新学习率
    # al()  # 设置模型为评估模式

    with torch.no_grad():
        total_correct = 0
        total_samples = 0

        for inputs, labels in val_dataloader:
            inputs,labels=inputs.long().to(device),labels.long().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f"Epoch {epoch+1}/{max_epochs}, Loss:{loss}, Validation Accuracy: {accuracy:.4f}")





