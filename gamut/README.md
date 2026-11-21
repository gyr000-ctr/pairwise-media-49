# gamut

## 题面（可直接粘贴）

色域映射。约束：按 OKLCh 裁剪并保持色相（误差 ≤2°）；亮度不变；已在色域内原样；越界亮度报错；10 万像素不超过 500 ms；并发安全。验收：cd gamut && python -m pytest tests -q -k gamut。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd gamut && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
