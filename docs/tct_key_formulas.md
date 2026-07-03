# TCT 方法部分关键公式

## 1. Attention-MIL 注意力池化

给定切片内有效 patch 集合 \(M\)，第 \(i\) 个 patch 的图像或特征为 \(x_i\)，编码与投影后的隐层表征为：

```math
h_i = f_\theta(x_i)
```

注意力分数与归一化注意力为：

```math
u_i = w_2^\top \tanh(W_1 h_i)
```

```math
a_i = \frac{\exp(u_i)}{\sum_{j \in M}\exp(u_j)}
```

切片级表征为：

```math
z = \sum_{i \in M} a_i h_i
```

## 2. TCT 与 HPV 多任务输出

TCT 分支输出 normal/abnormal 二分类概率，HPV 分支输出阴性、低风险、高风险三分类概率：

```math
p_{\mathrm{TCT}} = \operatorname{softmax}(W_T z + b_T)
```

```math
p_{\mathrm{HPV}} = \operatorname{softmax}(W_H z + b_H)
```

## 3. 类别加权交叉熵

类别 \(c\) 的权重由训练集类别计数得到：

```math
w_c = \frac{N}{C N_c}
```

其中，\(N\) 为训练样本总数，\(C\) 为类别数，\(N_c\) 为类别 \(c\) 的样本数。TCT 和 HPV 分类损失分别为：

```math
\mathcal{L}_{\mathrm{TCT}} = - w_y \log p_{\mathrm{TCT},y}
```

```math
\mathcal{L}_{\mathrm{HPV}} = - r w_g \log p_{\mathrm{HPV},g}
```

其中，\(y\) 为 TCT 标签，\(g\) 为 HPV 三分类标签，\(r\) 表示 HPV 标签是否可用。

## 4. ROI 检测分数聚合

设第 \(i\) 个 ROI 内的检测框集合为 \(B_i\)，检测框 \(b\) 的置信度为 \(c_b\)，则 ROI 检测证据定义为：

```math
s_{\max,i} = \max_{b \in B_i} c_b
```

```math
s_{\sum,i} = \sum_{b \in B_i} c_b
```

```math
n_i = |B_i|
```

## 5. 注意力引导损失与总目标

若 \(m_i\) 表示第 \(i\) 个 patch 是否包含异常细胞框或检测证据，则注意力引导目标为：

```math
q_i = \frac{m_i}{\sum_{j \in M} m_j}
```

对应的注意力引导损失为：

```math
\mathcal{L}_{guide} = - \sum_{i \in M} q_i \log(a_i)
```

总训练目标为：

```math
\mathcal{L} = \mathcal{L}_{\mathrm{TCT}} + \alpha \mathcal{L}_{\mathrm{HPV}} + \lambda \mathcal{L}_{guide}
```

当 \(\alpha=0\) 时，HPV 分支不参与训练；当 \(\lambda=0\) 时，细胞框或检测器输出不参与注意力约束。
