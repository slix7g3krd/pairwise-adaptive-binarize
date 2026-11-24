# pairwise-adaptive-binarize

## 领域

游戏、图形与多媒体

## 主题

扫描件自适应二值化

## 说明

窗口按字高、32 位积分图与背景归一。

初始状态下 `imaging/adaptive_binarize.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
