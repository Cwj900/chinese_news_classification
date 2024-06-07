import torch
import torch.nn as nn
import torch.optim as optim
from proprecessing import train_dataloader,val_dataloader


class CNN_LSTM(nn.Module):
    def __init__(self,num_classes,embedding_dim=100) :
        super(CNN_LSTM,self).__init__()
        #嵌入
        self.embedding = nn.Embedding(7000+2, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(64, 100, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(100, num_classes)


    def forward(self, inputs):
        embedded = self.embedding(inputs)
        embedded = embedded.permute(0, 2, 1)
        out = self.conv1(embedded)
        out = torch.relu(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = torch.relu(out)
        out = self.maxpool2(out)
        out, _ = self.lstm(out.transpose(1, 2))
        out = out[:, -1, :]  # 取最后一个时间步
        output = self.fc(out)
        return output
    
model = CNN_LSTM (15) # 创建模型实例
loss_fn = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 定义优化器

# 步骤3：训练模型
max_epochs = 10

for epoch in range(max_epochs):
    model.train()  # 设置模型为训练模式
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()  # 梯度清零

        # 前向传播
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    # 步骤4：评估模型
    model.eval()  # 设置模型为评估模式

    with torch.no_grad():
        total_correct = 0
        total_samples = 0

        for inputs, labels in val_dataloader:
            # 前向传播
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f"Epoch {epoch+1}/{max_epochs}, Validation Accuracy: {accuracy:.4f}")