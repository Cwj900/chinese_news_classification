import jieba
from tqdm import tqdm
from collections import Counter

#数据预处理：从数据集中提取文本和标签
class TextProcessor:
    def __init__(self, dataset_path= 'dataset/dataset.txt', stopwords_path = 'dataset/cn_stopwords.txt'):
        self.dataset_path = dataset_path
        self.stopwords = self.get_stopwords(stopwords_path)
        self.texts, self.labels = self.co_data()
        self.id2label, self.label2id = self.co_labeldict()
        self.labels_id = [self.label2id[label] for label in self.labels]

    #获取停用词
    def get_stopwords(self, stop_file_name):
        with open(stop_file_name, "r", encoding="utf-8") as file:
            lines = file.readlines()
        words = [i.strip() for i in lines]
        return words

    #提取文本
    def text_cleaning(self, text):
        text_result = ''
        for char in text:
            if '\u4e00' <= char <= '\u9fa5':
                text_result += char
        return text_result

    #提取分词后的文本和标签
    def co_data(self):
        labels = []
        texts = []

        with open(self.dataset_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                parts = line.split("_!_")
                labels.append(parts[2])
                texts.append(parts[3])

        # 对文本分词并去除停用词
        temp = texts.copy()
        texts = []
        for text in tqdm(temp,desc='分词并去除停用词'):
            result = self.text_cleaning(text)
            seg = jieba.cut(result, cut_all=False)
            text = [char for char in seg if char not in self.stopwords]
            texts.append(text)
        return texts, labels
    #构建标签对应的id
    def co_labeldict(self):
        label_freq = dict(Counter(self.labels))

        id2labels = {i: label for i, label in enumerate(label_freq)}
        labels2id = {label: i for i, label in enumerate(label_freq)}
        return id2labels, labels2id
    

