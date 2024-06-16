from sklearn.metrics import accuracy_score
from joblib import dump, load
import torch
import numpy as np
from proprocessml import X_test_ml,y_test_ml
from rank.proprecessing_rank import TextProcessor,DataProcessor
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 明确设置设备为CPU
device = torch.device("cpu")


class EnsembleModel:
    def __init__(self, deep_model, ml_model, threshold=0.4):
        self.deep_model = deep_model
        self.ml_model = ml_model
        self.threshold = threshold

    def predict(self, deep_inputs, ml_inputs):
        self.deep_model.eval()

        final_preds = []
        with torch.no_grad():
            for deep_input, ml_input in zip(deep_inputs, ml_inputs):
            # 将输入数据转换为张量，并确保它在CPU上
                deep_input = torch.tensor(deep_input, device=device).unsqueeze(0)
                deep_output = self.deep_model(deep_input)
                deep_probs = torch.softmax(deep_output, dim=1)
                deep_confidence, deep_pred = torch.max(deep_probs, dim=1)

                if deep_confidence.item() > self.threshold:
                    final_preds.append(deep_pred.item())
                else:
                    ml_pred = self.ml_model.predict(ml_input.reshape(1, -1))
                    final_preds.append(ml_pred[0])

        return np.array(final_preds)

    def score(self, deep_inputs, ml_inputs, y_true):
        y_pred = self.predict(deep_inputs, ml_inputs)
        return accuracy_score(y_true, y_pred)
class ModelEvaluator:
    def __init__(self, true_labels, pred_labels, id2label):
        self.true_labels = [id2label[true] for true in true_labels]
        self.pred_labels = [id2label[pred] for pred in pred_labels]
        self.labels_order = list(id2label.values())
    
    def plot_normalized_confusion_matrix(self):
        conf_matrix = confusion_matrix(self.true_labels, self.pred_labels, labels=self.labels_order)
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=self.labels_order, yticklabels=self.labels_order)
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def get_classification_report(self):
        report = classification_report(self.true_labels, self.pred_labels)
        return report

text_processor = TextProcessor()
texts = text_processor.texts
vocab = text_processor.vocab
label_ids = text_processor.labels_id
id2label = text_processor.id2label
label2id = text_processor.label2id
data_processor = DataProcessor(texts, label_ids, vocab)
test_vec = data_processor.test_data

deep_inputs = test_vec  # 是一个列表或者数组，每个元素是一个深度学习模型的输入
ml_inputs = X_test_ml  # 是一个列表或者数组，每个元素是一个机器学习模型的输入
y_true = y_test_ml  # 真实的标签
ml_model = load('ensemble_model.joblib')
deep_model = load('deep_model.joblib')
deep_model.eval()  # 确保模型在评估模式下
# 将模型转移到CPU
deep_model = deep_model.to(device)
ensemble = EnsembleModel(deep_model, ml_model)

# 使用集成模型进行预测
y_pred = ensemble.predict(deep_inputs, ml_inputs)

# 创建ModelEvaluator实例
evaluator = ModelEvaluator(y_true, y_pred, id2label)

# 绘制并显示归一化混淆矩阵
evaluator.plot_normalized_confusion_matrix()

# 打印分类报告
print(evaluator.get_classification_report())

# 使用集成模型进行评估
score = ensemble.score(deep_inputs, ml_inputs, y_true)
print('Accuracy:', score)

