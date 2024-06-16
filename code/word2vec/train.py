from preprocess import TextProcessor
from datasplit import DataProcessor

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from model import CNN,CNN_LSTM

def train(model,train_dataloader,val_dataloader,max_epochs=100):

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) 
    
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
        scheduler.step()  
    
        train_accuracy=eval(train_dataloader,model,device)
        val_accuracy=eval(val_dataloader,model,device)
        #寻找最佳模型
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
