# 用 2-2-1 网络学 XOR · 实测记录

**在线页面**：https://hhfree1.github.io/hhh-ai-xiaoku/

课程《人工智能通识》第 66 页「多层网络怎样学习参数？Python 实现与反向传播」的作业实现：
把页面给出的 XOR 示例**真实跑一遍**，把结果做成可核对的图表，部署到 GitHub Pages。

## 结论（全部来自实际运行）

| 输入 | 正确答案 | h1 | h2 | 概率 | 预测 |
|---|---|---|---|---|---|
| 0 0 | 0 | 0.030 | 0.030 | 0.0287 | 0 |
| 0 1 | 1 | 0.000 | 0.935 | 0.9745 | 1 |
| 1 0 | 1 | 0.935 | 0.000 | 0.9745 | 1 |
| 1 1 | 0 | 0.034 | 0.034 | 0.0305 | 0 |

- 损失：第 0 轮 `0.5620` → 第 300 轮 `0.0279`
- 四个样本全部分对，与课程脚本输出的 `All four XOR samples classified correctly.` 一致
- 补充实验：把初值换成随机数重复 20 次，**9 次**完全分对（45%）

## 复现

```bash
git clone https://github.com/hhfree1/hhh-ai-xiaoku.git
cd hhh-ai-xiaoku
pip install torch numpy matplotlib

python code/xor_pytorch.py    # 官方脚本原样运行
python code/run_and_log.py    # 同上逻辑 + 记录每 10 轮状态 → data/metrics.json
python code/make_figures.py   # 由 json 生成 figures/ 下四张图
python code/random_init.py    # 随机初值实验 → data/random_init.json
python code/build_page.py     # 由 json 重新生成 index.html
```

环境：Python 3.13.14 + PyTorch 2.15.0+cpu，Windows 11，CPU 训练约 7 秒（详见 `data/env.txt`）。
TensorFlow 版本未运行（本机未安装），`code/xor_tensorflow.py` 仅作对照保留。

## 目录

```
code/    官方脚本 2 个 + 记录/绘图/建页脚本 4 个
data/    metrics.json（每 10 轮状态）、final_predictions.csv、random_init.json、env.txt
figures/ 损失曲线、隐藏层激活、隐藏层空间、概率条形图
index.html  GitHub Pages 入口
```

## 固定超参（与第 66 页一致，未修改）

2 输入 → 2 个 sigmoid 隐藏单元 → 1 输出；四个样本整批；二元交叉熵（吃 logits）；
Adam，学习率 0.03；300 轮；float64；阈值 0.5。

## 这个实验不能证明什么

1. **不能证明泛化**：只有 4 个样本、没有测试集，拟合的就是这 4 个点。
2. **不能证明随机初值一定收敛**：官方初值已接近答案；换成随机初值后 20 次里只有 9 次成功，其余卡在局部极小（损失约 0.35～0.49）。
3. **不能证明多层网络万能**：通用逼近定理只保证「存在这样一组参数」，不保证梯度下降能找到。

## 代码来源

课程页 `assets/xor-code/` 下的 `xor_pytorch.py` 与 `xor_tensorflow.py`，原样保留、未改逻辑。
`run_and_log.py` 在其基础上增加日志与导出，计算部分完全一致。
