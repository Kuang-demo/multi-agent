# AI 安全基础概念与核心框架

AI 安全（AI Safety）是当前人工智能研究中最关键的领域之一。其核心目标是确保先进 AI 系统，特别是通用人工智能（AGI）和前沿大语言模型（Frontier LLM），在对齐、可控和可解释的前提下运行，避免对人类社会造成灾难性风险。

## 三大核心问题

AI 安全领域主要围绕三个核心问题展开。第一是对齐问题（Alignment Problem），即如何确保 AI 系统的目标与人类价值观一致。Stuart Russell 在 Human Compatible 一书中指出，直接给 AI 指定固定目标函数是危险的，因为超智能系统会以不可预知的方式追求目标，导致奖励破解（Reward Hacking）和工具收敛（Instrumental Convergence）。

第二是鲁棒性问题（Robustness），即 AI 系统在分布外（Out-of-Distribution, OOD）输入、对抗性攻击和长尾场景下是否依然可靠。OpenAI 的 GPT-4 系统卡展示了模型在越狱攻击（Jailbreak）下的脆弱性，单次对抗性提示就可能导致模型输出有害内容。

第三是可解释性问题（Interpretability），即人类能否理解模型的内部决策过程。Anthropic 在前沿可解释性研究中提出了机械论可解释性（Mechanistic Interpretability）范式，试图将神经网络的激活模式映射到人类可理解的概念。

## 主要研究机构与团队

全球 AI 安全研究呈现多中心格局。Anthropic 由前 OpenAI 员工 Dario Amodei 和 Daniela Amodei 于 2021 年创立，专注于构建安全的 AI 系统，开发了 Constitutional AI 方法。OpenAI 内部的 Superalignment 团队由 Ilya Sutskever 和 Jan Leike 共同领导，目标是在四年内解决超对齐问题，但该团队于 2024 年解散。DeepMind 的安全研究团队开发了 Sparrow 和 Gemini 的安全评估框架。此外，ARC Evals、METR（前身为 ARC Evals）和 Apollo Research 等独立机构专注于能力评估和风险评估。

## 关键评估框架

目前业界有几个重要的 AI 安全评估框架。NIST AI Risk Management Framework（AI RMF 1.0）由美国国家标准与技术研究院发布，提供 AI 系统的风险识别、测量和缓解指南。MLCommons 的 AI Safety Benchmark v0.5 包含 13 个危险类别，涵盖暴力、儿童安全、仇恨言论、自杀自残、化学武器、生物武器、放射性武器、网络安全、代码生成、隐私侵犯、心理健康、版权侵犯和诽谤。Anthropic 的 Responsible Scaling Policy（RSP）定义了 AI 安全等级（ASL-1 到 ASL-4），要求在每级部署更多安全措施。

## 安全等级分类

Anthropic 在 2023 年提出了 ASL（AI Safety Level）分类体系。ASL-1 对应当前未显示出显著危险能力的模型。ASL-2 对应在特定领域接近或超过人类专家水平的模型，需要部署监控和过滤系统。ASL-3 对应在多个危险领域超过人类专家水平的模型，需要隔离运行环境和严格的访问控制。ASL-4 对应通用超智能系统，需要数学上可证明的安全性保证。这一分类框架已被多家 AI 公司采纳为内部安全标准。

## 安全研究与能力研究的张力

AI 安全领域面临安全与能力之间的持续张力。当研究人员开发更安全的训练方法时，这些方法往往也会提升模型的一般能力。例如 RLHF（Reinforcement Learning from Human Feedback）既提高了模型的安全性，也显著提升了指令遵循能力。这种双重用途特性使得安全研究本身可能加速危险能力的发展，形成了安全研究的固有悖论。

## 中文 AI 安全社区

中文 AI 安全社区近年来迅速发展。智源研究院（BAAI）发布了悟道模型的安全评估报告。清华大学 CoAI 实验室在价值对齐和偏见检测方面有深入研究。上海期智研究院和上海人工智能实验室也设立了 AI 安全研究方向。奇安信和 360 等安全公司从网络安全角度切入 AI 安全，关注大模型的提示注入、越狱和隐私泄露问题。中国信通院发布了可信 AI 评估标准，涵盖鲁棒性、可解释性、公平性和隐私保护四个维度。
