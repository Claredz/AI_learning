# Data

这里存放实验使用的数据文件。

当前建议：
- 原始 CSV 放在 `data/raw/`
- 后续如果需要清洗后的数据，可以再增加别的子目录

当前示例数据字段：
- `height_cm`
- `weight_kg`
- `label_num`

可选字段：
- `label_text`

当前的 MLP 脚本只使用数值标签进行训练。
