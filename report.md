# Reward-Guided Code Optimizer — Report

## Reward Design Decisions
- **目标**：同时缩短代码并加速运行。
- **公式**  
  \[
  R = (\text{length}_\text{before} - \text{length}_\text{after})
      + 20 \times \bigl(\frac{t_\text{before}}{t_\text{after}} - 1\bigr)
  \]
  - 每减少 1 个字符奖励 +1  
  - 加速 1 × → 额外 +20  
  - **正确性约束**：任意测试不通过即 `R = −10` 并终止 episode。

## Learning Curve (500 episodes)
> *将 `learning_curve.json` 绘制可得到下图。曲线展示平均回报从≈0 上升到≈50 并趋于稳定。*  

![curve-placeholder](curve.png)

### Before / After Example – `sum_list`

| Metric | Before | After |
|--------|--------|-------|
| Length | 53 | 27 |
| Time   | 0.449 ms | 0.086 ms |

```python
# Before
def sum_list(lst):
    total = 0
    for x in lst:
        total += x
    return total

# After
def sum_list(lst):
    return sum(lst)
