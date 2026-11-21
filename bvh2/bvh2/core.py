"""bvh2 核心 API（初始未实现）。"""


class Bvh:
    def add(self, name, box):
        """加盒"""
        raise NotImplementedError

    def build(self):
        """建树"""
        raise NotImplementedError

    def nearest(self, origin, direction):
        """最近命中"""
        raise NotImplementedError
