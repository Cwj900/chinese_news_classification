from preprocess import TextProcessor
from text2vector import DataProcessor

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from gensim.models import Word2Vec

class CNN(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_dim, embedding_matrix ):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_layer.weight.data.copy_(embedding_matrix)
        self.embedding_layer.weight.requires_grad = True  # Allow embeddings to be fine-tuned
        self.conv1 = nn.Conv2d(1, 100, (3, embedding_dim))
        self.conv2 = nn.Conv2d(1, 100, (4, embedding_dim))
        self.conv3 = nn.Conv2d(1, 100, (5, embedding_dim))
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc = None


    def forward(self, inputs):
        out=self.embedding_layer(inputs)
        out = out.unsqueeze(1)
        out1 = self.conv1(out)
        out1=torch.squeeze(out1)
        out1 = torch.relu(out1)
        out1 = self.maxpool(out1)
        
        out2 = self.conv2(out)
        out2=torch.squeeze(out2)
        out2 = torch.relu(out2)
        out2 = self.maxpool(out2)

        out3 = self.conv3(out)
        out3=torch.squeeze(out3)
        out3 = torch.relu(out3)
        out3 = self.maxpool(out3)

        out1=torch.reshape(out1,(out1.shape[0],out1.shape[1]*out1.shape[2]))
        out2=torch.reshape(out2,(out2.shape[0],out2.shape[1]*out2.shape[2]))
        out3=torch.reshape(out3,(out3.shape[0],out3.shape[1]*out3.shape[2]))
        

        out = torch.cat((out1,out2,out3),1)

        if self.fc is None:
            self.fc = nn.Linear(out.shape[1], self.num_classes).to(out.device)
        output=self.fc(out)
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
vocab_size = len(word2vec_model.wv.key_to_index) + 1  #加1用于填充索引
embedding_dim=100
# 创建嵌入矩阵
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, index in word2vec_model.wv.key_to_index.items():
    embedding_matrix[index] = word2vec_model.wv[word]
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_matrix=embedding_matrix.to(device)

#初始化模型
model = CNN (15,vocab_size,100,embedding_matrix).to(device) # 创建模型实例
loss_fn = nn.CrossEntropyLoss().to(device)# 定义损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 定义优化器
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
    # 评估模型
    model.eval()  # 设置模型为评估模式

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



