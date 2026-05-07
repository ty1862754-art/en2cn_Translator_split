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
作业要求：作业里提到的“在验证过程中报告 BLEU 分数”就是指在这里算分并打印出来。
3. 测试集 (Test Set) —— 高考 / 最终大考
文件：test.txt
代码：在 chinese_bleu.ipynb 中。
逻辑：模型已经训练好了（拿到了 best_model.pt），此时我们用最严苛、完全没见过的测试集来给模型定性。
作业要求：最终报告里的 BLEU 分数和那几个翻译例子，必须取自测试集。