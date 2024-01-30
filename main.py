from model import forest
from preload import train_feat_matrix, train_label_vec, test_feat_matrix, test_label_vec
import numpy as np

# 用数据对树进行建模
forest.fit(train_feat_matrix, train_label_vec)

# 检测判断效果
prediction = forest.predict(test_feat_matrix)

# 打印预测结果
print(prediction)
print(f"we have {np.sum(prediction)} fake cases")