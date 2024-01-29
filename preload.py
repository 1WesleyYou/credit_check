import pandas as pd

# 列的编号，用于获取列内容
TIME = 0
AMOUNT = 29

data = pd.read_csv("dataset/creditcard.csv")
data = data.to_numpy()
# 特征矩阵：只包含数据，没有特征title 和标签tag
feat_matrix = data[:, :30]
label_vec = data[:, 30]

# print(feat_matrix[:, AMOUNT])
