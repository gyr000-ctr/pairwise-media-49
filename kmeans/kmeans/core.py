"""kmeans 核心 API（初始未实现）。"""


class KMeans:
    def __init__(self, k, seed=0):
        """构造"""
        raise NotImplementedError

    def fit(self, points):
        """训练"""
        raise NotImplementedError

    def centers(self):
        """中心"""
        raise NotImplementedError

    def error(self):
        """总误差"""
        raise NotImplementedError
