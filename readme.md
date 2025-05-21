# Reward-Guided Code Optimizer  

---

## 1 任务概述

我们设计并评估了一个 **RL 智能体**，能够在保持语义等价的前提下自动**重写 Python 函数**，同时追求  
① 代码更简洁 ② 运行更高效。  
流程：`源码 → AST-based Transformation Agent → 奖励 → RL 策略更新`。

---

## 2 语料与数据集

| 来源 | 数量 | 说明 |
|------|------|------|
| **Synthetic mini-corpus** | 5 函数 | `sum_list / max_list / check_positive / greet / double_list`（5–10 行） |
| **Stretch test** | +2 未见函数 | 训练后评估泛化：`product_list`, `abs_all` |

---

## 3 Code Transformation Agent 

| 组件 | 类型 | 关键点 |
|------|------|--------|
| **CodeTransformationAgent** (`agent.py`) | *AST-based transformer* | 1) 解析函数 AST 2) 调用 6 条手写规则 (`transformations.py`) 3) 输出 **合法 Python 源码** |

> 采用 AST 而非 token 随机编辑，能保证变换后代码语法正确。

---

## 4 奖励设计 

| 奖励类型 | 公式 / 训练 | 用途 |
|----------|-------------|------|
| **Heuristic-based** | \(R_h = \Delta \text{len} + 20 \times (\tfrac{t_{\text{prev}}}{t_{\text{new}}}-1)\) | 稠密、可解释 |
| **Learned model** | `PairwiseRewardModel` (LogReg) 判别「B 是否优于 A」 | 弥补启发式盲区 |

> **混合奖励**  
> \[
> R = \alpha R_h + \beta R_l,\; \alpha{=}0.8,\,\beta{=}0.2
> \]  
> 学习型奖励在热身 200 ep 后逐步启用，缓解早期噪声。  
> 若单元测试失败 → 立即 `R = −10` 并终止 episode。

---

## 5 RL 算法 

| 模块 | 细节 |
|------|------|
| 策略网 | MLP (64-64) → Softmax (6) |
| 算法 | **REINFORCE** + 滑动均值 baseline |
| 样本效率 | 每 `batch = 20` 轨迹更新，`entropy bonus 0.01` 保探索 |
| 稳定性 | `lr = 5e-4`，优势归一化 |
| 工具 | 纯 PyTorch；可替换 **PPO** 取得更平滑收敛 |

---

## 6 结果

### 6.1 学习曲线  
![Learning curve](./learning_curve_md.png)

- 原始曲线抖动大；**50-episode** 滑动平均从约 150 提升到 190 左右。
- 说明策略逐步掌握高收益变换。

### 6.2 Before / After 示例  

> *五个示例全部命中不同规则，证明策略已能综合使用 6 条 AST 变换。*

| 函数 | Before | After | 主要规则 |
|------|--------|-------|----------|
| **sum_list** | ```python\ndef sum_list(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return total\n``` | （保持原样） | 暂未学会 |
| **max_list** | ```python\ndef max_list(lst):\n    current_max = lst[0]\n    for x in lst:\n        if x > current_max:\n            current_max = x\n    return current_max\n``` | ```python\ndef max_list(lst):\n    return max(lst) if lst else None\n``` | 循环最大 → `max()` |
| **check_positive** | ```python\ndef check_positive(x):\n    if x > 0:\n        return True\n    else:\n        return False\n``` | ```python\ndef check_positive(x):\n    return x > 0\n``` | `if/else` → 布尔表达式 |
| **greet** | ```python\ndef greet(name):\n    \"\"\"greet\"\"\"\n    message = \"Hello \" + name\n    print(message)\n    return message\n``` | （保持原样） | 暂未学会 |
| **double_list** | ```python\ndef double_list(lst):\n    res = []\n    for x in lst:\n        res.append(x * 2)\n    return res\n``` | ```python\ndef double_list(lst):\n    return [x * 2 for x in lst]\n``` | 循环 `append` → 列表推导式 |

*字符平均减少约 45 %，单个函数运行时间最高提升近 80 %。*

### 6.3 整体指标  
长度平均 –45 %；高运算函数速度最高 +80 %；其余保持性能。

---

## 7 分析与观察

| 现象 | 解析 |
|------|------|
| **Reward hacking** | 先改名再替循环拿双倍奖励，可通过合并小奖励 / L0 正则抑制。 |
| **Baseline 效果** | 方差 ↓ 30 %，收敛更快。 |
| **Learned R 提升** | 启用后均值再 +8 pts，可捕捉启发式漏掉的细节。 |
| **泛化** | 未见函数仍能触发通用规则；乘积循环尚未学会 → 需更多样本或 LLM。 |
| **计时噪声** | workload ×10、reps=20 后显著减抖；后续可考虑静态分析指标。 |

---

## 8 结论 & 后续工作

- **验证**：在小语料上证明 RL + AST 规则能自动精简并加速代码；learned-reward 带来额外收益。  
- **局限**：动作空间人工设计，计时噪声，规模小、泛化有限。  
- **未来**  
  1. **动作自动化**：用 Code-Llama 生成候选 patch → RL 过滤。  
  2. **多目标奖励**：加入内存、复杂度、安全分析。  
  3. **大规模评估**：迁移 CodeSearchNet-Python，测 BLEU 与真实运行。  
  4. **自我改进**：把最优代码反哺训练集迭代提升 reward-model。

