# AI Learning Journal

这是一个用来记录我学习 AI 基础过程的仓库。

这里不是成熟框架，也不是“做完就结束”的项目。它更像一份持续更新的学习记录：我会在这里保存实验代码、学习笔记、实验观察和接下来的尝试方向。

我当前正在做的主线是：
- 用 `NumPy` 手写最小 MLP
- 一边写代码，一边理解前向传播、反向传播和训练过程
- 把实验结果和自己的理解一起记录下来

## 我在这里记录什么

- `experiments/`：动手验证想法的地方
- `notes/`：学习中的理解、困惑、推导和复盘
- `reports/`：每次实验之后留下的观察和总结
- `data/`：实验中使用的原始数据
- `figures/`：后续保存训练曲线或可视化结果

## 当前进展

现在这一步主要在做一个最小的二分类实验：
- 读取 CSV 数据
- 标准化特征
- 用一个隐藏层搭建最小 MLP
- 用 sigmoid 做激活
- 跑训练循环并观察 loss 和 accuracy

这部分的重点不是“做一个很完整的模型”，而是先把最基础的训练流程真正看懂。

## 目前可以从哪里开始看

如果想直接看当前主线，可以先从这些位置开始：

- `experiments/mlp/ann_binary_classifier.py`
- `experiments/mlp/generate_sample_data.py`
- `notes/learning_log.md`
- `reports/experiment_template.md`

## 如何运行当前实验

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 需要时重新生成示例数据

```bash
python experiments/mlp/generate_sample_data.py
```

3. 运行当前的 MLP 二分类实验

```bash
python experiments/mlp/ann_binary_classifier.py
```

当前示例数据位于 `data/raw/gender_height_weight_100.csv`，主要字段为：
- `height_cm`
- `weight_kg`
- `label_num`

其中 `label_text` 只是为了阅读更方便，训练脚本只使用数值标签。

## 接下来准备继续探索什么

- 比较不同激活函数带来的训练变化
- 比较不同损失函数的表现
- 把训练过程画成简单图像保存到 `figures/`

## 这个仓库的使用原则

- 先理解，再扩展
- 先记录过程，再追求整齐
- 保持轻量，不做过度工程化
- 让未来的自己还能看懂现在写下的东西
