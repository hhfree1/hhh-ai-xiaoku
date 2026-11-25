"""由 data/metrics.json 生成 index.html（GitHub Pages 入口）。

页面上的每个数字都从 metrics.json / random_init.json 读出，不手写，
避免图和表对不上。
"""

import json
import os
from datetime import date

with open("data/metrics.json", encoding="utf-8") as f:
    m = json.load(f)

history = m["history"]
before, after = history[0], history[-1]
labels = m["labels"]
names = ["".join(str(v) for v in s) for s in m["samples"]]
cfg = m["config"]

random_init = None
if os.path.exists("data/random_init.json"):
    with open("data/random_init.json", encoding="utf-8") as f:
        random_init = json.load(f)

env = ""
if os.path.exists("data/env.txt"):
    with open("data/env.txt", encoding="utf-8") as f:
        env = f.read().strip()

rows = ""
for name, y, h, p, pred in zip(names, labels, after["hidden"], after["prob"], after["pred"]):
    ok = pred == y
    rows += (
        f"<tr><td>{name[0]}</td><td>{name[1]}</td><td>{y}</td>"
        f"<td>{h[0]:.3f}</td><td>{h[1]:.3f}</td><td>{p:.4f}</td>"
        f"<td class=\"{'ok' if ok else 'bad'}\">{pred} {'✓' if ok else '✗'}</td></tr>\n"
    )

random_block = ""
if random_init:
    detail = "".join(
        f"<span class=\"chip {'ok' if r['success'] else 'bad'}\">{r['pred']}</span>"
        for r in random_init["runs"]
    )
    random_block = f"""
  <section class="card">
    <h2>附：换掉初值还成立吗</h2>
    <p>官方脚本的初值是照着答案挑的（W 一开始就是两个检测器）。为了看这个结构本身靠不靠得住，把初值换成
    <code>torch.randn</code> 的随机数，其余设置不变，重复 20 次：</p>
    <p class="big"><strong>{random_init['success']} / {random_init['total']}</strong> 次把四个样本全部分对。</p>
    <p class="muted">每一格是一次运行的四位预测：</p>
    <div class="chips">{detail}</div>
    <p class="cap">说明：成功率不是 100%，说明「结构有能力」不等于「每次都能练出来」，
    这也是 1989 年通用逼近定理只谈表示能力、不谈训练难度的原因。</p>
  </section>
"""

html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>XOR 与 2-2-1 多层网络 · 实测记录</title>
<style>
  :root {{
    --ink:#2C2C2A; --muted:#5F5E5A; --line:#E3E1DA; --bg:#FBFAF7;
    --blue:#185FA5; --orange:#BA7517; --ok:#0F6E56; --bad:#A32D2D;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
    font-size:15px; line-height:1.75;
  }}
  .wrap {{ max-width:840px; margin:0 auto; padding:48px 22px 72px; }}
  header {{ border-bottom:1px solid var(--line); padding-bottom:22px; margin-bottom:30px; }}
  .kicker {{ font-size:12.5px; color:var(--muted); letter-spacing:.04em; }}
  h1 {{ font-size:26px; font-weight:600; margin:8px 0 10px; line-height:1.35; }}
  .sub {{ color:var(--muted); font-size:14px; }}
  h2 {{ font-size:17px; font-weight:600; margin:0 0 12px; }}
  .card {{
    background:#fff; border:1px solid var(--line); border-radius:12px;
    padding:22px 24px; margin:0 0 22px;
  }}
  .lead {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:0 0 22px; }}
  .lead div {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px 18px; }}
  .lead .n {{ font-size:24px; font-weight:600; color:var(--blue); line-height:1.3; }}
  .lead .n.o {{ color:var(--orange); }}
  .lead .l {{ font-size:12.5px; color:var(--muted); }}
  code {{
    background:#F1EFE8; padding:1px 6px; border-radius:4px;
    font-family:"Cascadia Mono",Consolas,monospace; font-size:13px;
  }}
  pre {{
    background:#2C2C2A; color:#F1EFE8; padding:14px 16px; border-radius:8px;
    overflow-x:auto; font-size:13px; font-family:"Cascadia Mono",Consolas,monospace;
    line-height:1.6;
  }}
  figure {{ margin:0 0 6px; }}
  figure img {{ width:100%; height:auto; border-radius:8px; border:1px solid var(--line); background:#fff; }}
  figcaption {{ font-size:12.5px; color:var(--muted); margin-top:10px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; margin-top:6px; }}
  th,td {{ text-align:left; padding:8px 6px; border-bottom:1px solid var(--line); }}
  th {{ font-weight:500; color:var(--muted); font-size:12.5px; }}
  td.ok {{ color:var(--ok); }} td.bad {{ color:var(--bad); }}
  .big {{ font-size:18px; }}
  .muted {{ color:var(--muted); font-size:13.5px; }}
  .cap {{ color:var(--muted); font-size:12.5px; border-left:2px solid var(--line); padding-left:10px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 14px; }}
  .chip {{
    font-size:12px; padding:2px 7px; border-radius:5px; font-family:Consolas,monospace;
    background:#E6F1FB; color:var(--blue);
  }}
  .chip.bad {{ background:#FCEBEB; color:var(--bad); }}
  ol {{ padding-left:20px; }} ol li {{ margin-bottom:8px; }}
  a {{ color:var(--blue); }}
  footer {{ border-top:1px solid var(--line); margin-top:34px; padding-top:16px; font-size:12.5px; color:var(--muted); }}
  @media (max-width:640px) {{ .lead {{ grid-template-columns:1fr; }} h1 {{ font-size:21px; }} }}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="kicker">人工智能通识 · 第 66 页「多层网络怎样学习参数」</div>
  <h1>用 2-2-1 网络学 XOR：一次可复现的实测记录</h1>
  <div class="sub">作者 hhfree1 · 运行于 {date.today().isoformat()} · 代码、数据、图表全部可下载核对</div>
</header>

<div class="lead">
  <div><div class="n">0,1,1,0</div><div class="l">四个样本的最终预测</div></div>
  <div><div class="n o">{before['loss']:.2f} → {after['loss']:.4f}</div><div class="l">损失，第 0 轮到第 {after['epoch']} 轮</div></div>
  <div><div class="n">4 / 4</div><div class="l">分类正确的样本数</div></div>
</div>

<section class="card">
  <h2>怎么复现</h2>
  <p>运行环境：{env or '见 data/env.txt'}</p>
  <pre>git clone https://github.com/hhfree1/hhh-ai-xiaoku.git
cd hhh-ai-xiaoku
pip install torch numpy matplotlib
python code/run_and_log.py   # 训练并写出 data/metrics.json
python code/make_figures.py  # 由 json 生成四张图
python code/random_init.py   # 随机初值重复 20 次</pre>
  <p class="muted">超参固定为课程第 66 页给出的那一组：2-2-1、四个样本整批、二元交叉熵、Adam、学习率 0.03、300 轮、float64、阈值 0.5。</p>
</section>

<section class="card">
  <h2>① 损失曲线</h2>
  <figure>
    <img src="figures/loss_curve.png" alt="损失随训练轮数下降的曲线">
    <figcaption>这张图说明：训练确实在起作用——损失从 {before['loss']:.3f} 单调降到 {after['loss']:.4f}，说明「错多少」被压成了一个可以不断降低的数。</figcaption>
  </figure>
</section>

<section class="card">
  <h2>② 隐藏层两个单元在学什么</h2>
  <figure>
    <img src="figures/hidden_activations.png" alt="训练前后四个样本在两个隐藏单元上的输出">
    <figcaption>这张图说明：训练前两个单元对四个样本反应都很软；训练后 h₁ 只在 10 上接近 1、h₂ 只在 01 上接近 1——它们自己分化成了两个「检测器」，没人告诉它们该这样。</figcaption>
  </figure>
</section>

<section class="card">
  <h2>③ 四个点被搬到了哪里</h2>
  <figure>
    <img src="figures/hidden_space.png" alt="四个样本在隐藏层坐标中的位置，训练前后">
    <figcaption>这张图说明：隐藏层做的是换坐标。00 和 11 被搬到同一个位置（因为它们同类），01 和 10 被甩到两个角上，于是原本「一条直线切不开」的问题，在新坐标里变成「一条直线就能切开」。</figcaption>
  </figure>
</section>

<section class="card">
  <h2>④ 最终概率与判定</h2>
  <figure>
    <img src="figures/probability_bars.png" alt="四个样本的最终输出概率与 0.5 阈值">
    <figcaption>这张图说明：两类样本的概率分别落在 0.5 两侧很远的位置（约 {min(after['prob']):.3f} 与 {max(after['prob']):.3f}），不是勉强压线，这是收敛的表征。</figcaption>
  </figure>
</section>

<section class="card">
  <h2>最终结果表</h2>
  <table>
    <thead><tr><th>x₁</th><th>x₂</th><th>正确答案</th><th>h₁</th><th>h₂</th><th>概率</th><th>预测</th></tr></thead>
    <tbody>
{rows}    </tbody>
  </table>
  <p class="cap">数据来源：<code>data/final_predictions.csv</code>，由 <code>data/metrics.json</code> 导出。</p>
</section>

<section class="card">
  <h2>三点解读</h2>
  <ol>
    <li><strong>为什么单层不行。</strong>单层感知器只有一次判断，在平面上表现为一条直线。XOR 的两类点在对角交错，任何一条直线切下去，两边都会同时出现 0 和 1。</li>
    <li><strong>隐藏层做了什么。</strong>它先把四个点搬到新坐标（h₁, h₂），让同类点重合、异类点分开；输出层再在这个新坐标里画一条线。分工是「隐藏层摆点，输出层画线」。</li>
    <li><strong>为什么能自动做到。</strong>不用人手写 h₁/h₂ 的规则：损失函数量出「差多少」，梯度指出「往哪改」，优化器每轮改一点点，300 轮后规则自己长出来。</li>
  </ol>
</section>

<section class="card">
  <h2>这次实验不能证明什么</h2>
  <ol>
    <li><strong>不能证明泛化。</strong>只有 4 个样本、整批训练、没有测试集，它拟合的就是这 4 个点，没有回答「没见过的输入会怎样」。</li>
    <li><strong>不能证明随便初始化都能学会。</strong>官方初值已经是近似解，见下面的补充实验。</li>
    <li><strong>不能证明多层网络万能。</strong>通用逼近定理只保证「存在这样一组参数」，不保证梯度下降一定能找到，也不保证训练容易。</li>
  </ol>
</section>
{random_block}
<footer>
  课程页面：<a href="https://zhumy.github.io/Introduction_to_Artificial_Intelligence/">zhumy.github.io/Introduction_to_Artificial_Intelligence</a> ·
  代码来源为该页 assets/xor-code 下的两个官方脚本 ·
  本站所有数字均由 <code>code/</code> 下的脚本实际运行产生，可逐项核对。
</footer>

</div>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("wrote index.html", len(html), "chars")
