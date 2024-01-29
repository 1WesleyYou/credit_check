import pandas as pd
import numpy as np

data = pd.read_csv("dataset/creditcard.csv")
data = data.to_numpy()
# 特征矩阵：只包含数据，没有特征title 和标签tag
feat_matrix = data[:, 1:29]
