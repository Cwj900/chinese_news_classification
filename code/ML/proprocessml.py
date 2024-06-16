import jieba
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter  # 导入Counter类

# 获取停用词
def get_stopwords(stop_file_name):
    with open(stop_file_name, "r", encoding="utf-8") as file:
        lines = file.readlines()
    words = [i.strip() for i in lines]
    return words

# 字符清洗
def text_cleaning(text):
    text_result = ''
    for char in text:
        if '\u4e00' <= char <= '\u9fa5':
            text_result += char
    return text_result

# 数据预处理
def co_data(dataset_path, stopwords):
    labels = []
    texts = []

    with open(dataset_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in tqdm(lines):
            parts = line.strip().split("_!_")
            if len(parts) < 4:  # 确保数据格式正确
                continue
            labels.append(parts[2])
            cleaned_text = text_cleaning(parts[3])
            seg = jieba.cut(cleaned_text, cut_all=False)
            text = [char for char in seg if char not in stopwords]
            texts.append(' '.join(text))  # 用空格连接分词结果，以便CountVectorizer处理

    return texts, labels

# 整理类别和索引
def co_labeldict(labels):
    label_freq = Counter(labels)
    id2label = {i: label for i, label in enumerate(label_freq)}
    label2id = {label: i for i, label in enumerate(label_freq)}
    return id2label, label2id

# 加载停用词
stopwords = get_stopwords('./dataset/cn_stopwords.txt')

# 数据预处理
dataset_path = './dataset/dataset.txt'
texts, labels = co_data(dataset_path, stopwords)

# 创建标签索引映射
id2label, label2id = co_labeldict(labels)

# 使用CountVectorizer进行文本向量化
vectorizer = CountVectorizer(max_features=35000)

# 划分数据集
train_texts, rest_texts, train_labels, rest_labels = train_test_split(texts, labels, test_size=0.4, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(rest_texts, rest_labels, test_size=0.5, random_state=42)

# 向量化文本数据
X_train_ml = vectorizer.fit_transform(train_texts)
X_val_ml = vectorizer.transform(val_texts)
X_test_ml = vectorizer.transform(test_texts)

# 将标签转换为索引
y_train_ml = [label2id[label] for label in train_labels]
y_val_ml = [label2id[label] for label in val_labels]
y_test_ml = [label2id[label] for label in test_labels]
