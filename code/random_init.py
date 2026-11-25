"""随机初值实验：把参数初始化换成随机数，重复训练 20 次，统计成功率。

目的：官方脚本的初值是照着答案挑的（W 已经是两个检测器）。
这个实验回答「换掉这个起点，2-2-1 还能不能学会 XOR」。
"""

import json

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
target = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
labels = [0, 1, 1, 0]

results = []
for seed in range(20):
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(2, 2))
    b_hidden = torch.nn.Parameter(torch.zeros(2))
    v = torch.nn.Parameter(torch.randn(2, 1))
    b_output = torch.nn.Parameter(torch.zeros(1))
    params = [W, b_hidden, v, b_output]
    opt = torch.optim.Adam(params, lr=0.03)

    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        hidden = torch.sigmoid(x @ W + b_hidden)
        logits = hidden @ v + b_output
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        opt.step()

    with torch.no_grad():
        hidden = torch.sigmoid(x @ W + b_hidden)
        probs = torch.sigmoid(hidden @ v + b_output)
    preds = [int(p) for p in (probs >= 0.5).int().flatten().tolist()]
    results.append(
        {
            "seed": seed,
            "pred": preds,
            "prob": [round(float(p), 4) for p in probs.flatten().tolist()],
            "loss": round(float(loss.detach()), 5),
            "success": preds == labels,
        }
    )

ok = sum(r["success"] for r in results)
with open("data/random_init.json", "w", encoding="utf-8") as f:
    json.dump({"runs": results, "success": ok, "total": len(results)}, f, ensure_ascii=False, indent=1)

print(f"随机初值 20 次：{ok} 次完全分对")
for r in results:
    print(r["seed"], r["pred"], r["loss"], "OK" if r["success"] else "FAIL")
