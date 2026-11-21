"""beziersplit 核心 API（初始未实现）。"""


class BezierSplit:
    def length(self, p0, p1, p2, p3):
        """弧长"""
        raise NotImplementedError

    def t_at(self, p0, p1, p2, p3, s):
        """按弧长反解"""
        raise NotImplementedError
