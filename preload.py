import pandas as pd
from sklearn.model_selection import train_test_split  # 用于分裂训练集和测试集

# 列的编号，用于获取列内容
TIME = 0
AMOUNT = 29

data = pd.read_csv("dataset/creditcard.csv")
data = data.to_numpy()
# 特征矩阵：只包含数据，没有特征title 和标签tag
feat_matrix = data[:, :30]
label_vec = data[:, 30]

# print(feat_matrix[:, AMOUNT])
train_feat_matrix, test_feat_matrix, train_label_vec, test_label_vec \
    = train_test_split(feat_matrix, label_vec, test_size=0.2, random_state=42)  # 随机数种子

# print(test_label_vec.flatten())
