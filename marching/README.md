# marching

## 题面（可直接粘贴）

Marching Squares 等值线。约束：返回闭合折线；鞍点两种连接都能正确判定（不得穿越实心区）；空场返回空；边界单元处理正确；1 万单元不超过 400 ms；非法输入报错。验收：cd marching && python -m pytest tests -q -k contour。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd marching && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
