import torch
import torch.nn as nn

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






