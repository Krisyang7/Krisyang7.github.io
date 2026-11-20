---
title: "重思大模型 On-Policy Distillation：何时有效、为什么有效、如何修复失败"
description: 一篇从成功条件、token 级机制到实践方案的 OPD 论文精读，重点解释“强教师为什么也会教不会”。
publishDate: 2026-11-20
tags:
  - On-Policy Distillation
  - 知识蒸馏
  - 大语言模型
  - 论文阅读
noteType: paper
readingStatus: finished
paperTitle: "Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe"
authors:
  - Yaxuan Li
  - Yuxin Zuo
  - Bingxiang He
  - Jinqian Zhang
  - Chaojun Xiao
  - Cheng Qian
  - Tianyu Yu
  - Huan-ang Gao
  - Wenkai Yang
  - Zhiyuan Liu
  - Ning Ding
paperYear: 2026
venue: "arXiv:2604.13016v2"
paperUrl: https://arxiv.org/pdf/2604.13016
draft: false
featured: true
isExample: false
---

> **一句话结论：** OPD 是否有效，不由教师的参数量或 benchmark 分数单独决定，而取决于两件事：师生是否具有兼容的局部高概率分布，以及教师是否真的拥有学生尚未学到的能力。成功训练的直接机制，是在学生自己访问的状态上不断对齐师生共享的高概率 token。

这篇笔记只依据论文 v2 的[官方 PDF](https://arxiv.org/pdf/2604.13016)和[作者代码仓库](https://github.com/thunlp/OPD)整理。`[论文]` 表示作者报告的结论，`[分析]` 表示我基于论文证据做的归纳，`[假设]` 表示仍需实验验证的方向。这里的“thinking pattern”沿用论文术语，具体指师生在学生访问状态上的 top-$k$ token 分布兼容性，不把它扩大解释成完整的语义推理风格。

## 1. 现有方法及其缺陷：为什么还要重新研究 OPD？

### 从 off-policy 蒸馏到 on-policy 蒸馏

传统 off-policy Knowledge Distillation（KD）或 Supervised Fine-Tuning（SFT）让学生学习教师预先生成的固定序列。它的优点是简单稳定，但存在 **train–inference mismatch**：训练时学生沿着教师轨迹学习，推理时却必须沿自己的分布继续生成。一旦早期生成偏离教师轨迹，误差会在长序列中累积，这就是 exposure bias。[论文：PDF pp. 2, 18]

On-Policy Distillation（OPD）把监督位置换到了学生自己访问的状态：先由学生生成 rollout，再让教师对每个学生前缀给出逐 token 概率。这样既能缓解 exposure bias，又能获得比 outcome reward 更稠密的监督。

但论文指出，OPD 仍有一个此前没有被解释清楚的失败模式：**更大、benchmark 分数更高的教师，可能完全不能提升学生；相反，一个更弱但分布更兼容的教师反而能教得更好。** 因此，教师“知道得多”与学生“学得会”不是同一件事。[论文：PDF pp. 2, 5–8]

### 三种常见 OPD 实现也各有代价

| 方法 | 做法 | 优点 | 局限 |
|---|---|---|---|
| Sampled-Token OPD | 每个位置只使用学生实际采样的 token | 计算最轻；是 token-level reverse KL 的无偏单样本估计 | 单次监督不全面，低概率 token 的修正较慢；详见下方补充笔记 |
| Full-Vocabulary OPD | 每个位置计算全词表 KL | 监督最完整、梯度最密 | 内存开销为 $O(BTM)$，随 batch、长度和词表规模增长 |
| Top-$k$ OPD | 只在学生的 top-$k$ token 上计算重归一化 KL | 保留多 token 监督，同时降低教师查询成本 | 丢弃集合外概率质量，只是 full-vocabulary KL 的近似 |

**我的补充笔记：Sampled-Token OPD 的局限**

1. **监督信号不全面：** 单次采样只提供一个 Token 的误差信息，无法直接指导 Student 应该提高哪些 Token 的概率、降低哪些 Token 的概率，缺少完整的相对调整信息。
2. **低概率 Token 难以纠正：** 例如 Teacher 给“香蕉”35%，Student 仅给 5%，那么 Student 采到“香蕉”的概率很低，导致相关监督信号出现频率低、修正缓慢，需要更多采样才能充分学习。


所以，这篇论文真正追问的不是“OPD 能不能工作”，而是：

> **学生现在只会做自己熟悉的那几道题。老师把每道题“每个选项该打几分”的完整答案都给了它。那么在学生熟悉的这几道题上，老师这份详细答案什么时候真能帮学生提高？如果帮不上，是因为老师自己水平不行、是因为老师想教的东西学生根本碰不到、还是因为学生学习的方法（怎么用这份答案）有问题？**

## 2. 文章的方法：从目标函数到 token 级机制

### 2.1 OPD 的基本目标

给定 prompt $x$，学生策略 $\pi_\theta$ 生成 rollout
$\hat y=(\hat y_1,\ldots,\hat y_T)\sim\pi_\theta(\cdot\mid x)$。在学生已经生成的前缀 $(x,\hat y_{<t})$ 上，定义：

$$
p_t(v)=\pi_\theta(v\mid x,\hat y_{<t}),\qquad
q_t(v)=\pi_T(v\mid x,\hat y_{<t}),
$$

其中 $p_t$ 是学生的 next-token 分布，$q_t$ 是教师在**同一个学生前缀**上的分布。

标准 OPD 最小化学生到教师的 reverse KL 再利用自回归分解，它可以精确写成学生访问状态上的 token-level KL：

$$
\mathcal L_{\mathrm{OPD}}(\theta)
=\mathbb E_{x,\hat y\sim\pi_\theta}
\left[
\sum_{t=1}^{T}D_{\mathrm{KL}}(p_t\Vert q_t)
\right].
$$


Top-$k$ OPD 取学生高概率集合 $S_t=\operatorname{TopK}(p_t,k)$，先在集合内重归一化：

$$
\bar p_t^{(S_t)}(v)
=\frac{p_t(v)\mathbf 1[v\in S_t]}{\sum_{u\in S_t}p_t(u)},
\qquad
\bar q_t^{(S_t)}(v)
=\frac{q_t(v)\mathbf 1[v\in S_t]}{\sum_{u\in S_t}q_t(u)},
$$

再最小化 $D_{\mathrm{KL}}(\bar p_t^{(S_t)}\Vert\bar q_t^{(S_t)})$。论文的大部分机制实验使用这一形式。[论文：PDF p. 4，Eq. 5]

### 2.2 三个动态指标：判断教师信号是否真的可学

令学生和教师的 top-$k$ 集合分别为
$S_t^{(p)}=\operatorname{TopK}(p_t,k)$ 与
$S_t^{(q)}=\operatorname{TopK}(q_t,k)$。

**1）Overlap Ratio：师生是否在关注相同候选。**

$$
\mathcal M_{\mathrm{overlap}}
=\mathbb E_t\left[
\frac{|S_t^{(p)}\cap S_t^{(q)}|}{k}
\right].
$$

值越高，说明学生越接近教师的高概率 support region；低值表示两者存在明显 mode mismatch。[论文：PDF p. 4，Eq. 6]

**2）Overlap-Token Advantage：共享 Token 内的概率是否一致。**

**设计目的：** Teacher 和 Student 即使选中了相同的 Token，也不代表它们分配的概率相同。因此，作者先取双方 Top-k Token 的交集，并在交集内重新归一化，得到 $\bar p_t$（Student）和 $\bar q_t$（Teacher），比较双方对共享 Token 的概率分配是否一致。

### ① 单个 Token 的指标：$A_t(v)$

$$
A_t(v)=\bar p_t(v)
\bigl(\log\bar q_t(v)-\log\bar p_t(v)\bigr)
$$

**设计思路：** 先计算 Teacher 与 Student 对某个 Token 的对数概率差异，再乘以 Student 的概率，表示该 Token 对整体差异的贡献。

**单个 Token 的数值含义：**

| $A_t(v)$ | 概率关系 | 说明 |
|---|---|---|
| 正数 | $\bar q_t(v)>\bar p_t(v)$ | Teacher 概率更高，Student 相对低估该 Token |
| 负数 | $\bar p_t(v)>\bar q_t(v)$ | Student 概率更高，Student 相对高估该 Token |
| 等于 0 | $\bar p_t(v)=\bar q_t(v)$ | 双方对该 Token 的概率一致 |

**注意：** 单个 $A_t(v)$ 可以为正，也可以为负，用来判断 Student 对某个共享 Token 是相对高估还是低估。

### ② 整体指标：$\mathcal M_{\mathrm{adv}}$

$$
\mathcal M_{\mathrm{adv}}
=\mathbb E_t\left[
\frac{1}{|S_t^{(p)}\cap S_t^{(q)}|}
\sum_{v\in S_t^{(p)}\cap S_t^{(q)}}A_t(v)
\right]
$$

**设计思路：**

- $\sum A_t(v)$：将所有共享 Token 的贡献相加，等于**负的 Reverse KL**，衡量共享集合内部的整体概率分布差异。
- $\frac{1}{|S_t^{(p)}\cap S_t^{(q)}|}$：除以交集大小，计算每个共享 Token 的平均贡献。
- $\mathbb E_t$：对不同生成位置求平均，得到整体指标。

**整体指标的数值含义：**

| $\mathcal M_{\mathrm{adv}}$ | 说明 |
|---|---|
| 等于 0 | 共享集合内 Teacher 和 Student 的概率分布完全一致 |
| 接近 0（如 -0.01） | 共享集合内的概率分配差异较小 |
| 负值较大（如 -0.5） | 共享集合内的概率分配差异较大 |
| 大于 0 | 理论上不可能（按上述定义且交集非空） |

**注意：** 整体指标为负，并不代表 Student 对所有 Token 都过度自信，而是说明双方概率分布存在差异。因为有的 Token 被高估，有的被低估，汇总后等于负的 Reverse KL，因此整体指标始终小于或等于 0。

### 总结

- **单个 $A_t(v)$：** 看 Student 对某个共享 Token 是高估还是低估，通过正负号判断。
- **整体 $\mathcal M_{\mathrm{adv}}$：** 看 Student 和 Teacher 在共享 Token 内的概率分布有多一致，越接近 0 越一致，越负则差异越大。

**局限性：** 整体指标会受到交集大小的影响。如果交集只有一个 Token，重新归一化后指标必然为 0，即使原始概率不同。因此，需要结合交集大小和 Token 重合率一起分析。

[论文：PDF p. 5，Eq. 7]

**3）Entropy Gap：师生置信度结构是否一致。**

$$
\Delta H_t=|H(q_t)-H(p_t)|.
$$

趋近 0 表示两者在相同学生状态上的不确定性结构逐渐匹配。[论文：PDF p. 5，Eq. 8]

### 2.3 方法的完整逻辑

论文先用对照实验和 reverse distillation 识别两个成功条件：

1. **Thinking-pattern consistency**：学生和教师的局部高概率分布必须兼容；
2. **Genuinely new capabilities**：教师必须提供学生训练中尚未见过的能力，不能只是规模更大或分数更高。

随后，论文在 token 层面比较成功与失败 run，并做 overlap-only / non-overlap-only 干预。最后给出两条恢复 recipe：

- **Off-policy cold start**：先用教师生成的 rollout 对学生做一段 SFT，再切换到 OPD；
- **Teacher-aligned prompt selection**：使用与教师 post-training 更一致的 prompt 模板或内容进行 OPD。

前者从模型初始化侧缩小分布差距，后者从数据侧让学生更常访问教师熟悉的状态。[论文：PDF pp. 5–14]

## 3. 本文解决了什么问题，为什么这些方法有效？

### 3.1 论文实际解决的四个问题

**第一，解释“强教师为什么也会教不会”。**

[论文] 教师的参数规模或 benchmark 分数不是 OPD 可学性的充分条件。若师生 thinking pattern 不兼容，学生一开始就很少访问教师支持的高概率区域，后续训练也难以追回早期损失；若教师与学生来自同一训练 pipeline、没有额外新能力，即使分数更高，也可能没有新的可迁移信息。[论文：PDF pp. 5–8，Figs. 2–5]

**第二，定位 OPD 真正起作用的 token 区域。**

[论文] 成功 OPD 中，师生共享的 top-$k$ token 数量并不大，却承载双方 **97%–99%** 的总概率质量。只在 overlap token 上优化，几乎可以匹配标准 Student Top-$k$ OPD；只优化 non-overlap token 则明显更弱。[论文：PDF pp. 9–10, 25，Figs. 6–7, 18]

**第三，把相关性推进到干预证据。**

[论文] 在成功 run 中，overlap ratio 大约从 **72% 上升到 91% 以上**，advantage 趋近 0，entropy gap 收窄；失败 run 的三个指标从一开始就停滞。更重要的是，overlap-only 消融直接保留了几乎全部收益，因此共享高概率区域不仅“伴随成功”，也是主要优化位置。[论文：PDF pp. 9–11，Figs. 6–7]

**第四，给出可以修复失败训练的方案。**

[论文] Cold start 先用 200K 教师响应做 SFT，再在约 30K 去重 prompts 上运行 OPD，使学生以更高初始 overlap、更小 entropy gap 开始训练；prompt 模板对齐提高 overlap，prompt 内容对齐则让更多概率质量集中到更少的共享高概率 token 上。[论文：PDF pp. 11–14, 27，Figs. 8–10]

### 3.2 为什么方法可以解决：一条完整因果链

可以把论文的核心机制压缩为：

[论文 + 分析] 论文实验证明了序列层面的全局信号仍有信息、局部训练却可能失败；“不同位置的梯度方向不一致并相互抵消”是作者尚未验证的解释性假设。

```text
thinking pattern 不兼容
→ 学生访问的状态远离教师熟悉区域
→ 师生共享的高概率 token 少，或共享区域内概率分配不一致
→ 教师信号即使在序列层面与正确性相关，也未必能在学生当前策略附近形成可利用的局部梯度
→ overlap 停滞，OPD 失败
```

反过来，cold start 或 teacher-aligned prompts 先把学生带入教师熟悉的局部区域。进入这一共享区域后，reverse KL 会把更多概率集中到教师偏好的共同 token 上，并逐步挤出竞争 token：

```text
共享 token 被教师偏好
→ 学生把更多概率放到共享 token
→ 它们更稳定地留在学生 top-k 中
→ 后续梯度继续作用于同一共享区域
→ overlap 自增强，训练持续推进
```

[分析] 因此，OPD 的主要作用不是把学生突然拉向它此前完全不考虑的 token，而是在学生已经能够访问、师生共同认可的高概率候选中重新分配概率。这也解释了为什么初始分布兼容性如此重要。[论文：PDF pp. 9–11]

### 3.3 关键证据如何闭环

| 结论 | 关键证据 | 能支持什么 |
|---|---|---|
| Thinking pattern 比教师分数更关键 | Qwen 对照中，GRPO 教师的初始 overlap 更高，蒸馏效果更好；教师自身 benchmark 并非全面更高 | 教师分数不能单独预测 OPD 成败 |
| 新能力比单纯规模更关键 | DeepSeek 家族的 gap recovery 为 16.9% vs. 5.3%；Qwen 家族为 58.6% vs. 15.6%，额外 RL 教师更可迁移 | 教师需要学生未见的新能力，而非只是更大 |
| OPD 会学习教师的局部分布 | Reverse distillation 中，1.5B 与 7B 同家族教师把学生拉回近似相同水平 | 同家族不同规模教师在学生访问状态上可能局部分布近似 |
| 共享高概率 token 是主要作用位置 | Overlap-only 几乎匹配 Student Top-$k$，non-overlap 显著更弱 | overlap 区域具有干预意义，不只是相关性 |
| 全局有信息不等于局部可优化 | 成功和失败教师区分正确/错误 rollout 的 AUROC 分别约 0.73、0.75 | 失败不能简单归因于全局 reward 质量更差 |

[论文：PDF pp. 6–10, 16，Figs. 2–7, 14]

这里仍要保留一个边界：两条 recipe 主要修复 **thinking-pattern gap**，不能凭空给教师创造“新知识”。如果教师没有学生尚未获得的能力，分布对齐本身仍不足以产生显著增益。

## 4. 现有方法的弊端与未来方向

### 4.1 作者明确指出的局限

**1）证据范围局限于数学推理。**

论文的模型、数据和评测集中在 AIME、AMC 等数学任务，尚不能直接推断到代码生成、开放式任务或多轮 agent。[论文：PDF p. 19]

**2）“新知识”的来源仍未被完全隔离。**

论文用额外 RL 后的教师作为新能力来源，但 pre-training 数据差异、tokenizer、架构、post-training 数据与目标可能纠缠。作者指出，受控 pre-training ablation 成本很高，这仍是开放问题。[论文：PDF p. 19]

**3）在本文设置中，长轨迹会削弱稠密监督的可靠性。**

3K/7K response length 附近效果较好，10K/15K 在训练后期出现 entropy 与 gradient norm 尖峰；教师从学生 prefix 继续生成的准确率优势，从 1K prefix 的约 **+0.37** 降到 16K prefix 的约 **+0.02**。学生走得越远，前缀越可能离开教师熟悉状态。[论文：PDF pp. 14–16，Figs. 11–13]

**4）Prompt 对齐可能压缩探索能力。**

使用教师 post-training prompts 会显著降低学生 entropy。作者建议混入教师分布之外的 prompts，避免过度压缩学生的探索空间。[论文：PDF pp. 13–14，Fig. 10]

**5）局部梯度几何仍只是一个假设。**

失败教师的全局 reward 仍能区分正确与错误 rollout，但有效 gradient norm 更小。作者提出 per-token advantage 可能在不同位置方向不一致、聚合时相互抵消，但尚未直接验证。[论文：PDF pp. 16–17]

**6）Self-distillation 仍需单独验证。**

单模型自蒸馏天然具有更高 pattern consistency，新能力往往来自 privileged information。论文的两条件框架能否直接迁移到该 setting，作者认为仍是下一步问题。[论文：PDF p. 19]

### 4.2 作者提出的未来方向

- 在代码、开放式生成和其他非数学领域验证“两条件 + 高概率 token 对齐”是否仍成立；
- 研究 pre-training 数据对 OPD 可蒸馏性的独立影响；
- 扩展到 self-distillation，分析 privileged information 如何构成新能力；
- 对长程推理采用混合方案：短 horizon 使用稠密 token 监督，长 horizon 使用稀疏 outcome reward；或用 curriculum 逐步延长监督长度；
- 直接分析 per-token 梯度方向，设计能利用 anisotropic reward structure 的新目标。

[论文：PDF pp. 16–17, 19]

### 4.3 基于论文证据，我认为值得继续验证的方向

下面不是作者已经证明的结论，而是从现有证据推出的研究假设。

| 观察 | `[假设]` 可验证方向 | 最小验证方式 |
|---|---|---|
| 论文用 top-$k$ overlap 操作化 thinking pattern | 它预测的是局部分布兼容性，不一定等价于语义层面的推理风格 | 固定 tokenizer 和正确答案，构造不同推理路径，检查 overlap 是否仍能预测 OPD 成败 |
| 额外 RL 同时改变数据、目标和训练步数 | “新能力”的可迁移性可能来自其中某一因素，而非 RL 本身 | 固定基座与总预算，对数据、reward objective、steps 做正交消融 |
| Cold start 使用 200K 教师 rollout | 它有效，但 compute-normalized 性价比尚不清楚 | 固定教师 token、训练 FLOPs 和 wall-clock，搜索达到目标 overlap 的最低 cold-start 预算 |
| Overlap token 数量与 overlap probability mass 可能不一致 | 单独用 overlap ratio 选教师或 prompt 可能误判 | 联合 overlap mass、entropy gap 与 gradient coherence，预测最终 gap recovery |
| 证据集中在 Qwen/DeepSeek 衍生模型和数学任务 | 两个成功条件可能受 tokenizer、架构和任务类型影响 | 做跨 tokenizer、跨架构、代码与多轮 agent 的矩阵实验 |

我最想保留的研究问题是：**能否只用训练最初 10–20 步的 overlap mass、entropy gap 和 gradient coherence，提前判断某个 teacher–student pair 是否值得继续蒸馏？** 如果能，就可以在完整训练之前做一次低成本“可蒸馏性体检”，把论文的事后机制分析变成事前选择工具。
