import numpy as np
from preload import feat_matrix, label_vec, TIME, AMOUNT


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None  # 初始化树结构为无

    # 特征分类树的构造函数, x 是特征矩阵，y 是代入向量
    # 叶节点包含 "class" 标签，node 不包含
    def fit(self, x, y, depth=0):
        if depth == self.max_depth or len(set(y)) == 1:
            most_frequent_class = max(set(y), key=y.count)
            # 树到了最大深度或者只有一个属性，就到了叶子节点
            # key获取了列表中出现最多的元素
            return {"class": most_frequent_class}

        # 选择最佳分裂的特征和点
        best_feat, best_value = self._find_best_split(x, y)  # 函数定义在下面

        # 找不到能分解的
        if not best_feat:
            max_freq = max(set(y), key=y.count)
            return {"class": max_freq}

        # 分裂数据集
        left_indices = x[:, best_feat] <= best_value  # bool 数组, 保存的是符合的向量
        right_indices = ~left_indices  # 取反

        # 递归构建左右子树
        left_subtree = self.fit(x[left_indices], y[left_indices], depth + 1)
        right_subtree = self.fit(x[right_indices], y[right_indices], depth + 1)

        # 返回当前节点的信息
        return {
            "feature_index": best_feat,
            "threshold": best_value,
            "left": left_subtree,
            "right": right_subtree,
        }

    # 找到每个节点的最好切割点，也就是通过特征矩阵和实际向量获得最佳的分类阈值和特征
    # 用 _ 开头表示这个函数是 private 的, 这是写码习惯
    def _find_best_split(self, x, y):
        num_feature = x.shape(1)
        # 这里用信息增益算法判断最好切割点
        best_info_gain = -1
        best_feat = None
        best_value = 0

        current_entropy = self._entropy(y)

        for feature_index in range(num_feature):
            # 去除重复操作，输出按照升序排序
            unique_values = np.unique(x[:, feature_index])
            for value in unique_values:
                left_indices = x[:, feature_index] <= value
                right_indices = ~left_indices

                left_entropy = self._entropy(y[left_indices])
                right_entropy = self._entropy(y[right_indices])

                # 信息增益算法Gain(D,a)=Ent(D) - \Sigma\frac{ |D^v| }{ |D| } \times Ent(D^v)
                info_gain = current_entropy - (  # bool 数组的求和获得的是 True 的数量
                        np.sum(left_indices) / len(y) * left_entropy +
                        np.sum(right_indices) / len(y) * right_entropy
                )

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feat = feature_index
                    best_value = value
        return best_feat, best_value

    def _entropy(self, y):
        unique_class, class_counts, = np.unique(y, return_counts=True)
        probability = class_counts / len(y)
        entropy = -np.sum(probability * np.log2(probability))
        return entropy

    def predict(self, x):
        return np.array([self._predict_tree(x1, self.tree) for x1 in x])

    def _predict_tree(self, x, node):
        if "class" in node:
            # 叶子节点直接返回
            return node["class"]

        if x[node["feature_index"]] <= node["threshold"]:
            return self._predict_tree(x, node["left"])
        else:
            return self._predict_tree(x, node["right"])


class RandomForest:
    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, x, y):
        for _ in range(self.num_trees):
            # 随机选择一部分数据用于训练每棵树
            sample_indices = np.random.choice(len(x), len(x), replace=True)
            x_sampled = x[sample_indices]
            y_sampled = y[sample_indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(x_sampled, y_sampled)
            self.trees.append(tree)

    def predict(self, y):
        predictions = np.zeros((len(y), self.num_trees))

        for i, tree in enumerate(self.trees):
            # 树的输出应该是多个组成了向量
            predictions[:, i] = tree.predict(y)

        # 多数投票决定最终结果
        final_predictions = np.apply_along_axis(lambda x1: np.bincount(x1).argmax(), axis=1, arr=predictions)

        return final_predictions

# todo: 解释 y 的定义
