"""dft 核心 API（初始未实现）。"""


class Dft:
    def transform(self, samples):
        """变换"""
        raise NotImplementedError

    def window(self, n, kind='hann'):
        """窗函数"""
        raise NotImplementedError
