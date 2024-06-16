import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from .rank.proprecessing_rank import TextProcessor,DataProcessor
from word2vec.preprocess import TextProcessor,DataProcessor
from word2vec.model import TextCNN,CNN_LSTM
from rank.model_rank import TextCNN,CNN_LSTM
from rank.model_rank import TextCNN,CNN_LSTM
from gensim.models import Word2Vec
from joblib import dump


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


# #数据预处理
# text_processor = TextProcessor()
# texts = text_processor.texts
# label_ids = text_processor.labels_id
# id2label = text_processor.id2label

# #加载Word2Vec模型
# word2vec_model = Word2Vec.load("model/word2vec.model")
# #划分数据
# data_processor = DataProcessor(word2vec_model, texts, label_ids)
# train_dataloader = data_processor.train_loader
# val_dataloader = data_processor.val_loader

# #初始化嵌入层
# vocab_size = len(word2vec_model.wv.key_to_index) + 1 
# embedding_dim=100
# # 创建嵌入矩阵
# embedding_matrix = np.zeros((vocab_size, embedding_dim))
# for word, index in word2vec_model.wv.key_to_index.items():
#     embedding_matrix[index] = word2vec_model.wv[word]
# embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
# #构建模型
# cnn_model = TextCNN (15,vocab_size,embedding_dim,embedding_matrix)
# lstm_model = CNN_LSTM (15,vocab_size,embedding_dim,embedding_matrix) 
# #训练模型
# train(cnn_model,train_dataloader,val_dataloader)

#数据预处理
text_processor = TextProcessor()
texts = text_processor.texts
vocab = text_processor.vocab
label_ids = text_processor.labels_id
id2label = text_processor.id2label
label2id = text_processor.label2id
#划分数据
data_processor = DataProcessor(texts, label_ids, vocab)
train_dataloader = data_processor.train_loader
val_dataloader = data_processor.val_loader
test_dataloader = data_processor.test_loader
#构建模型
vocab_size=len(vocab)+2
model = TextCNN (15,vocab_size,100)
#训练模型
train(model, train_dataloader,val_dataloader)

# 保存模型
dump(model, 'deep_model.joblib')