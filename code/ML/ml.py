from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from proprocessml import X_train_ml,X_test_ml,X_val_ml,y_test_ml,y_train_ml,y_val_ml
from joblib import dump



class MachineLearningEnsemble(BaseEstimator):
    def __init__(self):
        # 初始化机器学习模型
        self.models = {
            'sgd': SGDClassifier(),
            'pa': PassiveAggressiveClassifier(),
            'svc': LinearSVC(),
            'ridge': RidgeClassifier(),
            'gb': GradientBoostingClassifier()
        }
        # 初始化投票分类器
        self.voting_classifier = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='hard'
        )       
        
    def fit(self, X, y):
        # 训练每个独立的机器学习模型
        for model in self.models.values():
            model.fit(X, y)
        
        # 训练投票分类器
        self.voting_classifier.fit(X, y)
        
    def predict(self, X):
        # 使用投票分类器进行预测
        return self.voting_classifier.predict(X)
    
    def score(self, X, y):
        # 评估投票分类器的准确率
        return accuracy_score(y, self.predict(X))
    
# 创建集成模型的实例
ensemble_model = MachineLearningEnsemble()
# 训练集成模型
ensemble_model.fit(X_train_ml, y_train_ml)

# 在验证集上评估集成模型的性能
val_accuracy = ensemble_model.score(X_val_ml, y_val_ml)
print(f"Validation Accuracy: {val_accuracy:.4f}")
# 如果你想评估每个单独模型的性能
for name, model in ensemble_model.models.items():
    model_accuracy = model.score(X_val_ml, y_val_ml)
    print(f"{name} Validation Accuracy: {model_accuracy:.4f}")

# 在测试集上评估集成模型的性能
test_accuracy = ensemble_model.score(X_test_ml, y_test_ml)
print(f"Test Accuracy: {test_accuracy:.4f}")
# 保存模型
dump(ensemble_model, 'ensemble_model.joblib')