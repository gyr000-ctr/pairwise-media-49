# texwarp

## 题面（可直接粘贴）

噪声域扭曲。约束：扭曲幅度有上界且零幅度时恒等；必须用多倍频（octaves）分形噪声；同种子可复现、不同种子不同；扭曲后 UV 保持有限且幅度 ≤ 设定值；非法幅度/倍频数可区分报错；10 万次不超过 300 ms；并发安全。验收：cd texwarp && python -m pytest tests -q -k warp。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd texwarp && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
