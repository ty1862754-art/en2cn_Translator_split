1.1补全代码后
1.2模型训练
先做小数据冒烟训练
在 translator_en2cn.py 保持 DEBUG = True，然后运行：
# 1. 进入项目根目录
cd E:\code\DP_final

# 2. 进入 runtime_core 目录（数据路径是相对路径，必须在这里运行）
cd "Assignment en2cn_Translator_split\runtime_core"

# 3. 激活 conda 环境
conda activate cse5023_py312

# 4. 运行脚本
python translator_en2cn.py
这会用 train_mini/dev_mini，通常几秒到几分钟能跑完。


检查环境：
conda deactivate
conda activate cse5023_py312
python -c "import sys; print(sys.executable)"
python -m pip show nltk



conda activate pytorch_env
cd d:\code\en2cn_Translator_split\runtime_core
python translator_en2cn.py




正式从零训练
把 translator_en2cn.py 里的 DEBUG 改成 False，再运行：
python translator_en2cn.py
这会切到 train/dev 全量数据。





从零开始训练你的Transformer模型：你已经有了 translator_en2cn.py 并成功跑出了模型。
绘制训练损失曲线：目录下已经有保存下来的 loss_curve.png 和 loss_history.json。
在合适的轮次间隔保存中间模型：你的代码中已经实现了保存 checkpoint.pt 和 best_model.pt 的逻辑。
在 chinese_bleu.ipynb 中实现 BLEU 评估及分词：这个文件目前已经修改完毕并处于就绪状态，包含了基于 jieba 的分词和基于 evaluate 的 BLEU 计算。
❌ 还没做完、亟待解决的部分（你的下一步工作）
1. “在验证过程中报告BLEU分数” —— 【代码层面未完成】 目前你的 translator_en2cn.py 里面的 run_validation 函数，仅仅是打印了 4 个句子的预测结果，并没有在训练时自动计算和打印验证集的 BLEU 分数。你需要在验证逻辑里引入 BLEU 的计算。

2. “在测试集上评估最终模型的性能” —— 【执行层面未完成】 虽然你前面目录下有一个 predictions_test.txt，但来路不明。为了严谨，你最好：

写一个专门的预测代码（利用你跑出来的 best_model.pt），将测试集真正翻译一遍生成新的 predictions_test.txt。
打开并运行 docs_explainer/chinese_bleu.ipynb，得出最终的那个 BLEU 数值。
3. “在报告中列举若干测试集的翻译输出示例...” —— 【文档层面未完成】 这部分是文字工作，需要你在最终提交的 Word/PDF 报告中，挑几个模型翻译的句子进行点评


有了模型，在inpy中使用
分词处理数据测试集
你不仅需要在验证过程中报告BLEU分数，还需要在测试集上评估最终模型的性能
此外，请在报告中列举若干测试集的翻译输出示例，具体说明模型的性能表现




第一步：开始训练模型
打开终端，进入 runtime_core 目录，运行训练脚本。

bash
python translator_en2cn.py
注意：如果你想快点看到结果，可以将脚本顶部的 DEBUG 设置为 True（但这只训练很小的数据集，正式作业建议用 False）。
产出：训练结束后，你会在 save/models/ 目录下看到 model_epoch_05.pt, best_model.pt, loss_curve.png 等文件。
第二步：确认训练损失 (Loss)
检查 save/models/loss_curve.png。

确认曲线是平滑下降的。
这张图要保存好，它是你报告里的第一个重要素材。

第三步：编写翻译推理脚本 (Inference)
你需要一个脚本来加载保存的模型，并把测试集 (data/en-cn/test.txt) 的英文翻译成中文。 你可以在 chinese_bleu.ipynb 中添加一个单元格来做这件事，或者新建一个脚本。代码逻辑如下：

加载模型权重（例如 best_model.pt）。
读取测试集中的英文句子。
使用 greedy_decode 函数生成中文。
将结果保存为 predictions.txt。

第四步：对翻译结果进行分词
在 chinese_bleu.ipynb 中，使用你刚才问的那段 jieba 代码：

将 input_file 设置为你的翻译结果 predictions.txt。
运行代码，生成带空格的 predictions_segmented.txt。
同样的方法：对测试集的“标准答案”也运行一遍分词，生成 reference_segmented.txt。

第五步：计算 BLEU 分数
在 chinese_bleu.ipynb 中，使用 evaluate 库计算分数：

python
import evaluate
bleu = evaluate.load("bleu")
# 读取分词后的结果
with open("predictions_segmented.txt", "r", encoding="utf-8") as f:
    preds = [line.strip() for line in f]
with open("reference_segmented.txt", "r", encoding="utf-8") as f:
    refs = [[line.strip()] for line in f] # 注意 refs 是嵌套列表
results = bleu.compute(predictions=preds, references=refs)
print(results)


第六步：对比不同轮次（可选但推荐）
为了体现你完成了“保存中间模型”的要求，你可以分别对 model_epoch_05.pt、model_epoch_10.pt 和 best_model.pt 重复第三到第五步。

目标：展示随着训练轮数增加，BLEU 分数是如何上升的。



第七步：整理测试集示例并写报告
从测试集的评估结果中，人工挑选 3-5 个典型的翻译例子：

例子 1 (翻译得好)：展示模型学到了复杂的句式。
例子 2 (翻译得不好)：分析为什么错了（例如：人名翻译错、漏词、语序不对）。
结论：总结你的模型目前能达到什么水平，还有哪些提升空间。




 训练集 (Train Set) —— 课本
文件：train.txt
代码：在 translator_en2cn.py 的主循环里。
逻辑：模型反复看这些句子，计算误差并调整自己的权重。
2. 验证集 (Validation Set) —— 模拟考
文件：dev.txt
代码：在 translator_en2cn.py 的 run_validation 函数里。
逻辑：模型练了几轮后，我们用它没见过的验证集考考它。如果你发现验证集分数不再上升，说明模型练到头了。
作业要求：作业里提到的"在验证过程中报告 BLEU 分数"就是指在这里算分并打印出来。
3. 测试集 (Test Set) —— 高考 / 最终大考
文件：test.txt
代码：在 chinese_bleu.ipynb 中。
逻辑：模型已经训练好了（拿到了 best_model.pt），此时我们用最严苛、完全没见过的测试集来给模型定性。
作业要求：最终报告里的 BLEU 分数和那几个翻译例子，必须取自测试集。

================================================================================
1.3 预热策略与学习率调优 — 执行计划
================================================================================

【目标】使用 Cosine Annealing with Warm-up 调度器替代固定学习率，通过调优提
升 BLEU 分数。

【基线参考】固定 lr=1e-4，测试集 BLEU = 0.2121

================================================================================
步骤一：理解策略原理
================================================================================
- 预热阶段：学习率从 0 线性上升至峰值 lr（warmup_epochs 轮完成）
- 退火阶段：预热结束后，学习率按半余弦曲线从峰值 lr 平滑下降至 0
- 公式：lr = peak_lr * 0.5 * (1 + cos(π * progress))

================================================================================
步骤二：代码实现
================================================================================
新建文件：runtime_core/translator_en2cn_cosine.py
（从 translator_en2cn.py 复制后修改）

改动点：
  A. 文件头部新增 import
     from torch.optim.lr_scheduler import LambdaLR
     import math

  B. get_config 新增参数
     'peak_lr': 1e-4,           # 峰值学习率（可调）
     'warmup_epochs': 5,        # 预热轮次（可调）
     'experiment_name': 'cosine_warmup_lr1e-4_warm5',

  C. 定义调度器（optimizer 之后）
     def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
         def lr_lambda(current_step):
             if current_step < warmup_steps:
                 return float(current_step) / float(max(1, warmup_steps))
             progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
             return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
         return LambdaLR(optimizer, lr_lambda)

     warmup_steps = config['warmup_epochs'] * len(data.train_data)
     total_steps = config['num_epochs'] * len(data.train_data)
     scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

  D. 训练循环（optimizer.step() 之后加入）
     scheduler.step()

  E. 输出路径隔离
      所有模型文件存入 save/models-cosine/{experiment_name}/
      与基线（save/models/）完全隔离

================================================================================
步骤三：参数实验与调优
================================================================================
实验设计（共 2 组，与基线对比）：

  实验名                          peak_lr   warmup    验证 BLEU
  ─────────────────────────────────────────────────────────────
  基线（已有，固定 lr）             1e-4       无       0.2121
  cosine_warmup_lr1e-4_warm5     1e-4       5 轮      待测
  cosine_warmup_lr5e-5_warm3     5e-5       3 轮      待测

运行命令：
  cd d:\code\en2cn_Translator_split\runtime_core
  conda activate pytorch_env
  python translator_en2cn_cosine.py

每次运行前修改：
  1. get_config 中的 peak_lr / warmup_epochs
  2. experiment_name 与参数保持一致

================================================================================
步骤四：结合 BLEU 评估（用验证集快速对比）
================================================================================
- 每组实验训练结束后，直接读取控制台最后打印的 VALIDATION BLEU SCORE
- 验证 BLEU 更高的参数即为更优配置
- 不要用测试集调参，测试集只留到最后用一次

调参流程：
  实验A ──→ 验证 BLEU = ?    ──┐
                                ├── 对比筛选
  实验B ──→ 验证 BLEU = ?    ──┘
                                ↓
                        锁定最佳参数
                                ↓
                 只用一次测试集 → 最终 BLEU 写入报告

================================================================================
步骤五：报告记录
================================================================================
在 assignment.tex 的 1.3 节写入：
  1. 学习率策略的数学原理解释
  2. 对比表格（基线 vs 余弦实验）
  3. 最佳参数组合及其 BLEU
  4. 简短结论（哪些参数最优，为什么）

================================================================================
时间预估
================================================================================
  操作                              预估时间
  ──────────────────────────────────────────────
  创建 translator_en2cn_cosine.py    10 分钟
  运行实验A（训练 50 epoch）         1~2 小时
  运行实验B（训练 50 epoch）         1~2 小时
  对比验证 BLEU + 选最佳参数         5 分钟
  最佳参数跑测试集 BLEU              ~10 分钟
  写入报告                          ~15 分钟
================================================================================

================================================================================
1.4 超参数消融实验 — 执行计划
================================================================================

【目标】探究模型架构超参数（n_layer, h_num, d_model）对翻译性能的影响，通过
控制变量法逐一消融，以表格形式记录结论。

【基线模型】(实验B) n_layer=6, h_num=8, d_model=256, d_ff=1024, peak_lr=3e-4, warmup=5
  基线测试集 BLEU = 0.2323

================================================================================
步骤一：确立统一基线
================================================================================
- 基线 = 1.3 最优结果（实验B: peak_lr=3e-4, warmup=5, n_layer=6, h_num=8, d_model=256）
- 后续所有消融实验仅改变目标架构参数，训练策略保持不变
- 每组训练 50 epoch，使用相同随机种子环境

================================================================================
步骤二：明确消融变量
================================================================================
三个独立变量，分别消融：

  变量          基线值    消融值    说明
  ─────────────────────────────────────────
  n_layer        6         3       Transformer层数减半
  h_num          8         16      注意力头数翻倍
  d_model        256       128     嵌入维度减半 (d_ff同步: 1024→512)

每次只改变一个变量，其余保持基线值不变。

================================================================================
步骤三：控制变量实验设计
================================================================================

  实验名                  n_layer   h_num   d_model   d_ff    预估参数量
  ─────────────────────────────────────────────────────────────────
  基线（实验B）              6        8       256      1024     ~9.4M
  ablation_nlayer_3         3        8       256      1024     ~5.3M
  ablation_hnum_16          6       16       256      1024     ~9.4M
  ablation_dmodel_128       6        8       128      512      ~3.5M

运行命令：
  cd d:\code\en2cn_Translator_split\runtime_core
  conda activate pytorch_env
  python translator_en2cn_ablation.py --n_layer 3 --name ablation_nlayer_3
  python translator_en2cn_ablation.py --h_num 16 --name ablation_hnum_16
  python translator_en2cn_ablation.py --d_model 128 --name ablation_dmodel_128

输出路径：save/models-ablation/{name}/

================================================================================
步骤四：表格化记录与报告
================================================================================
在 assignment.tex 的 1.4 节写入：
  1. 消融实验总表（模型配置 + BLEU）
  2. 各变量的影响规律分析
  3. 模型规模与性能的权衡结论

================================================================================
时间预估
================================================================================
  操作                              预估时间
  ──────────────────────────────────────────────
  创建 translator_en2cn_ablation.py  10 分钟
  运行消融实验1（n_layer=3）         1~1.5 小时
  运行消融实验2（h_num=16）          1~1.5 小时
  运行消融实验3（d_model=128）       1~1.5 小时
  汇总对比 + 更新报告                ~15 分钟
================================================================================

================================================================================
1.5 位置嵌入 — 执行计划
================================================================================

【目标】实现可学习位置嵌入（Learnable Positional Embedding），替换原有的正弦/余弦
绝对位置编码，对比两种位置嵌入策略对翻译性能的影响。

【基线模型】(实验B) n_layer=6, h_num=8, d_model=256, d_ff=1024, peak_lr=3e-4, warmup=5
  基线测试集 BLEU = 0.2323

================================================================================
步骤一：实现可学习位置嵌入
================================================================================
修改位置编码方式：
  原有方式：正弦/余弦绝对位置编码（Sinusoidal PE）- 固定不变，不可学习
  新方式：可学习位置嵌入（Learnable PE）- 通过反向传播自动更新

代码修改位置：tokenization.py 或模型Embedding层
  - 创建可学习的位置参数矩阵 pos_embed，形状为 (max_len, d_model)
  - 在forward中，将词嵌入与位置嵌入相加后送入Transformer

================================================================================
步骤二：对比实验设计
================================================================================

  实验名                    位置嵌入方式     说明
  ──────────────────────────────────────────────────────
  基线（实验B）              Sinusoidal PE    固定正弦/余弦编码
  learnable_pe              Learnable PE     可学习位置嵌入

运行命令：
  cd d:\code\en2cn_Translator_split\runtime_core
  conda activate pytorch_env
  python translator_en2cn_pos_embed.py --pe_type learnable

输出路径：save/models-posembed/{name}/

================================================================================
步骤三：性能评估与对比
================================================================================
记录指标：
  - 训练损失曲线（loss_curve.png）
  - 验证集 BLEU（每10轮打印）
  - 测试集 BLEU（最终评估）
  - 收敛速度对比

================================================================================
步骤四：报告撰写与深度讨论
================================================================================
在 assignment.tex 的 1.5 节写入：
  1. 位置嵌入策略的原理（正弦/余弦 vs 可学习）
  2. 对比表格（基线 vs 可学习位置嵌入）
  3. 实验发现（收敛速度、性能指标变化）
  4. 深度讨论：
     - 不同位置嵌入策略的作用机制
     - 为什么可学习位置嵌入会带来某种性能差异
     - 各自在Transformer架构中的角色

================================================================================
时间预估
================================================================================
  操作                              预估时间
  ──────────────────────────────────────────────
  修改位置嵌入代码                  15 分钟
  运行基线对比实验（Sinusoidal）     1~2 小时
  运行可学习PE实验（Learnable）      1~2 小时
  汇总对比 + 更新报告                ~20 分钟
================================================================================

【实验结果】

实验名                    位置嵌入方式    测试集 BLEU    最终 Loss    额外参数量
──────────────────────────────────────────────────────────────────────────
基线（实验B）              Sinusoidal PE   0.2323         0.084        0
learnable_pe              Learnable PE    0.2180         0.084        30,720

【结论】
- 可学习位置嵌入 BLEU 略低于正弦编码（-1.43 个百分点）
- 训练 Loss 几乎完全一致，说明拟合能力无差异
- 正弦编码的固定位置偏置起到了隐式正则化作用
- 在大规模数据场景下可学习嵌入可能有优势，中等规模任务中正弦编码更稳健

【生成文件】
- runtime_core/translator_en2cn_pos_embed.py    训练脚本
- runtime_core/eval_pos_embed.py                评估脚本
- save/models-posembed/learnable_pe/            模型文件目录
================================================================================