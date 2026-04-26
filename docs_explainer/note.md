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

判断是否“从零开始”
你当前这版脚本没有加载 checkpoint 的逻辑，默认每次都是随机初始化训练，也就是从零开始。
保存目录是：Assignment en2cn_Translator_split/runtime_core/save/models