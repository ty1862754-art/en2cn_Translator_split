# Runtime Core 项目结构

```
runtime_core/
│
├── chinese_bleu.ipynb              # BLEU 评估 Notebook（推理 + 分词 + 算分）
├── environment.txt                  # 环境配置说明
├── tokenization.py                  # 数据预处理（分词、建词典、batch padding）
├── translator_en2cn.py              # 基线训练脚本（固定学习率）
├── translator_en2cn_cosine.py       # 改进训练脚本（Cosine Warmup 调度器）
│
├── data/
│   └── en-cn/
│       ├── all_data.txt             # 全量数据汇总
│       ├── train.txt                # 训练集
│       ├── dev.txt                  # 验证集
│       ├── test.txt                 # 测试集
│       ├── train_mini.txt           # 训练集（小规模，调试用）
│       ├── dev_mini.txt             # 验证集（小规模，调试用）
│       └── test_mini.txt            # 测试集（小规模，调试用）
│
├── model/
│   └── transformer.py               # Transformer 模型架构
│
└── save/
    ├── models/                      # 基线实验输出（固定 lr=1e-4）
    │   ├── best_model.pt           # 最佳模型权重
    │   ├── checkpoint.pt           # 完整检查点（含优化器状态）
    │   ├── model.pt                # 最终模型权重
    │   ├── model_epoch_05.pt       # epoch 5 中间模型
    │   ├── model_epoch_10.pt       # epoch 10 中间模型
    │   ├── model_epoch_20.pt       # epoch 20 中间模型
    │   ├── model_epoch_30.pt       # epoch 30 中间模型
    │   ├── model_epoch_40.pt       # epoch 40 中间模型
    │   ├── loss_history.json       # 训练损失历史
    │   └── loss_curve.png          # 训练损失曲线图
    │
    ├── models-cosine/               # 余弦实验输出（运行时自动创建）
    │   └── cosine_warmup_lr*_warm*/ # 每组实验一个子目录
    │       ├── config.json          # 实验配置记录
    │       ├── best_model.pt
    │       ├── checkpoint.pt
    │       ├── loss_history.json
    │       ├── loss_curve.png
    │       └── model_epoch_*.pt
    │
    └── predictions_segmented.txt    # 测试集翻译结果（已分词）
```
