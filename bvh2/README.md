# bvh2

## 题面（可直接粘贴）

场景 BVH 最近命中。约束：最近命中与暴力遍历一致；射线与盒求交正确；空场景报错；1 万盒子建树不超过 500 ms；并发查询安全。验收：cd bvh2 && python -m pytest tests -q -k bvh。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd bvh2 && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
