# RLHF 与对齐技术详解

人类反馈强化学习（Reinforcement Learning from Human Feedback, RLHF）是当前大语言模型对齐的核心技术。该方法由 OpenAI 在 InstructGPT 论文（2022）中首次系统化提出，随后被 Anthropic、DeepMind 和 Meta 广泛采用。

## RLHF 三阶段流程

RLHF 的完整流程分为三个独立阶段。第一阶段是监督微调（Supervised Fine-Tuning, SFT），使用人工标注的高质量指令-回复对来训练基础模型，使模型初步学会遵循人类指令。标注者根据给定的提示编写理想的回复，数据量通常在数万到数十万条。研究表明 SFT 阶段的数据质量远比数据量重要，高质量的一万条数据可能优于低质量的十万条数据。

第二阶段是奖励建模（Reward Modeling, RM）。收集人类对同一提示的多个模型回复进行偏好排序，训练一个奖励模型来预测人类偏好。具体实现使用 Bradley-Terry 模型：给定提示 x 和两个回复 y_a 和 y_b，人类偏好 y_a 优于 y_b 的概率由两个回复的奖励值差决定。奖励模型通常与基础模型使用相同架构，但将最后的语言模型头替换为标量输出头。训练损失函数为交叉熵损失，目标是最大化人类偏好选择的概率。

第三阶段是近端策略优化（Proximal Policy Optimization, PPO）。使用奖励模型作为奖励信号，通过 PPO 算法优化语言模型策略。PPO 的核心创新是使用剪裁替代目标（Clipped Surrogate Objective），限制策略更新的幅度，防止模型在单次优化步骤中偏离太远。同时加入 KL 散度惩罚项，防止优化后的策略与原始 SFT 策略差异过大，避免模型遗忘通用语言能力。整个 RLHF 过程需要对上述三个阶段迭代改进，因为策略变化后之前收集的偏好数据可能不再适用。

## 主要对齐方法的进展

直接偏好优化（Direct Preference Optimization, DPO）由 Stanford 在 2023 年提出，是 RLHF 的最重要改进。DPO 的关键洞察是：奖励模型的最优解可以表示为策略概率和参考策略概率之比的函数。因此可以不显式训练独立的奖励模型，直接将偏好数据转化为策略的优化信号，避免了奖励建模的不稳定性。Llama 3 和 Qwen 2 等新一代开源模型均采用了 DPO 训练。

Constitutional AI（CAI）由 Anthropic 在 2022 年底提出。该方法使用一组自然语言编写的宪法规则（Constitution）来替代人类反馈。宪法包含从联合国人权宣言、苹果服务条款、Google AI 原则等多个来源提取的伦理准则。训练过程分为监督阶段和强化学习阶段。在监督阶段，模型通过自我批判和修订来改进有害回复。在强化学习阶段，使用 AI 反馈代替人类反馈来计算偏好。Claude 系列模型是 Constitutional AI 方法的最著名应用，其安全性和有用性在 LMSYS 排行榜上长期排名前列。

RLHF 变体包括：Kahneman-Tversky Optimization (KTO)，该方法将偏好数据转化为 Kahneman-Tversky 价值函数，处理不平衡偏好数据；Identity Preference Optimization (IPO)，解决 DPO 可能过度拟合偏好的问题；Group Relative Policy Optimization (GRPO)，DeepSeek 提出的基于组的相对策略优化方法，去除了价值函数模型以降低显存开销。

## 价值对齐的数据构建

对齐数据是 RLHF 成功的基础。OpenAI 的 InstructGPT 使用了约 13K SFT 数据和 33K 偏好比较数据。Anthropic 的 HH-RLHF 数据集包含约 170K 条人类关于有用性和无害性的偏好比较。构建高质量对齐数据面临可扩展监督（Scalable Oversight）挑战：随着模型能力提升，人类标注者越来越难以评判模型输出。对此出现了多种解决方案：辩论式对齐（Debate），让两个 AI 互相辩论而人类做最终裁定；递归奖励建模（Recursive Reward Modeling），用 AI 辅助人类评估更复杂任务；基于过程的反馈（Process-Based Feedback），从最终答案的评判转向推理步骤的评判。

## 过度对齐与对齐税

过度对齐（Over-alignment）是指安全训练过度导致模型拒绝合理的请求，即安全性与有用性的权衡（Safety-Usefulness Trade-off）。研究表明高风险类别的拒绝率可以达到 95% 以上，但低风险类别的合理拒绝率应控制在 5% 以内。对齐税（Alignment Tax）指对齐训练后模型在某些学术基准上的性能下降，通常在 1-3 个百分点。通过改进对齐方法缩小对齐税是当前研究热点。
