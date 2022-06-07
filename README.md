<h1 align="center">Prompt-Tuning</h1>

+ A pipeline for Prompt-tuning
+ 集成主流的Prompt-tuning相关方法，以及search template策略
+ 提供Prompt-tuning完整的执行pipeline

# Requirements
本项目相关的依赖包参考requirements.txt，也可直接使用如下指令安装：
```shell
pip install -r requirements.txt
```

# Usage
+ core下放置相关prompt-tuning模型
+ core/gen_template下是相关template生成方法，执行入口参考run_gen_template.py，执行示例如下：
```python
python3 run_gen_template.py \
    --task_name CoLA \
    --k 16 \
    --dev_rate 1 \
    --data_loader glue \
    --template_generator lm_bff \
    --data_dir data/original/CoLA \
    --output_dir data/output \
    --generator_config_path data/config/lm_bff.json
```
+ 模型实现放在core目录下，执行入口参考run_prompt_tuning.py，执行示例如下：
```python
python3 run_prompt_tuning.py \
    --data_dir data/CoLA/ \
    --do_train \
    --do_eval \
    --do_predict \
    --model_name_or_path bert \
    --num_k 16 \
    --max_steps 1000 \
    --eval_steps 100 \
    --learning_rate 1e-5 \
    --output_dir result/ \
    --seed 16
    --template "*cls**sent_0*_It_was*mask*.*sep+*" \
    --mapping "{'0':'terrible','1':'great'}" \
    --num_sample 16 \
```
+ data放置相关config及datasets，由于数据集比较庞大，可使用scripts下的下载脚本自行下载，如下：
```shell
cd data
sh download_clue_dataset.sh
sh download_glue_dataset.sh
```
+ tools放置相关工具方法及数据集处理方法等

# Paper
更详细的论文解读和阅读笔记 ☞  [点这里](https://github.com/DengBoCong/nlp-paper)

+ [Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://arxiv.org/pdf/2001.07676.pdf)
+ [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://arxiv.org/pdf/2010.15980.pdf)
+ [Making Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/pdf/2012.15723.pdf)
+ [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/pdf/2101.00190.pdf)
+ [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf)
+ [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf)
+ [Noisy Channel Language Model Prompting for Few-Shot Text Classification](https://arxiv.org/pdf/2108.04106.pdf)
+ [PPT: Pre-trained Prompt Tuning for Few-shot Learning](https://arxiv.org/pdf/2109.04332.pdf)
+ [SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer](https://arxiv.org/pdf/2110.07904.pdf)

# Reference
+ https://github.com/princeton-nlp/LM-BFF
+ https://github.com/shmsw25/Channel-LM-Prompting

# Dataset
+ GLUE：https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar
```shell
sh scripts/download_glue_dataset.sh
```
+ CLUE：https://github.com/CLUEbenchmark/CLUE
```shell
sh scripts/download_clue_dataset.sh
```



