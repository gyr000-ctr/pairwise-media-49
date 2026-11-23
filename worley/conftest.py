import builtins
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import worley

# 测试体内的占位 smoke 用例以裸名字 `Worley` 引用构造函数，
# 这里经 builtins 暴露，使其在不改动既有测试的前提下可解析。
if not hasattr(builtins, "Worley"):
    builtins.Worley = worley.Worley


import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    # 给本套件全部用例加上 "worley" 关键字，
    # 使验收命令 `python -m pytest tests -q -k worley` 能匹配到用例。
    for item in items:
        item.extra_keyword_matches.add("worley")
