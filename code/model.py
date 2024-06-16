import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from proprecessing import TextProcessor,DataProcessor


class CNN_LSTM(nn.Module):
    def __init__(self,num_classes,vocabulary_num,embedding_dim=100) :
        super(CNN_LSTM,self).__init__()
        #嵌入
        self.embedding = nn.Embedding(vocabulary_num, embedding_dim)
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
    
class TextCNN(nn.Module):
    def __init__(self,num_classes,vocab_size,embedding_dim) :
        super(TextCNN,self).__init__()
        #嵌入
        self.num_classes=num_classes
        self.embedding_layer=nn.Embedding(vocab_size, embedding_dim)
        # self.embedding_layer.weight.data.copy_(embedding_matrix)
        self.conv1 = nn.Conv2d(1, 100, (3,100))
        self.conv2 = nn.Conv2d(1, 100, (4,100))
        self.conv3 = nn.Conv2d(1, 100, (5,100))
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        # self.lstm = nn.LSTM(64, 100, dropout=0.2, batch_first=True)
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

def train(model,train_dataloader,val_dataloader):

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device) 
    loss_fn = nn.CrossEntropyLoss().to(device)  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 优化器

    # 训练模型
    max_epochs = 10
    best_val_accuracy=0
    best_model=None

    for epoch in range(max_epochs):
        model.train()  
        for inputs, labels in tqdm(train_dataloader):
            inputs,labels=inputs.long().to(device),labels.long().to(device)
            optimizer.zero_grad() 

            # 前向传播
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()
    
        train_accuracy=eval(train_dataloader,model,device)
        val_accuracy=eval(val_dataloader,model,device)

        if val_accuracy>best_val_accuracy:
            best_model=model
        print(f"Epoch {epoch+1}/{max_epochs}, Loss：{loss}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        # 保存整个模型
    torch.save(best_model, './model/model_complete.pth')



def eval(val_dataloader,model,device):
    # 评估模型
    model.eval()  # 设置模型为评估模式

    with torch.no_grad():
        total_correct = 0
        total_samples = 0

        for inputs, labels in val_dataloader:
            inputs,labels=inputs.long().to(device),labels.long().to(device)
            # 前向传播
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples
    return accuracy


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

vocab_size=len(vocab)+2
model = TextCNN (15,vocab_size,100)
train(model, train_dataloader,val_dataloader)
