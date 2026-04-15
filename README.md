# Awesome Ascend Skills

基于华为昇腾 NPU 开发的 AI Agent 知识库，以 Skill 形式组织，支持 Claude Code、OpenCode、Cursor、Trae、Codex 等多种 AI 编程工具。

---

## 目录

- [简介](#简介)
- [快速开始](#快速开始)
- [安装指南](#安装指南)
- [Skill 导航](#skill-导航)
- [外部 Skills](#外部-skills-external-skills)
- [Skill 工作原理](#skill-工作原理)
- [治理规范](#治理规范)
- [贡献指南](#贡献指南)
- [提交 PR](#提交-pr)
- [官方文档](#官方文档)
- [许可证](#许可证)

---

## 简介

**Awesome Ascend Skills** 是一套面向华为昇腾 NPU 开发的结构化知识库。仓库同时包含：

- 单个 skill：解决单一问题，如 `npu-smi`、`hccl-test`
- 领域技能包：聚合某个方向的多个子 skill，如 `mindspeed-llm-skills`
- 官方推荐安装包：面向新人和常见任务场景的分层 bundle，如 `ascend-base`、`ascend-inference`
- 外部同步 skills：从外部 Ascend skill 仓库自动同步的扩展能力

如果你是第一次使用，建议不要直接从平铺 skill 列表开始看，而是先看下面的 `快速开始` 和 `安装指南`。

---

## 快速开始

### 我应该先装什么？

```text
Start
├─ 你是第一次使用，或者还不确定该装什么？
│  ├─ Yes → 先安装 `ascend-base`
│  └─ No  → 进入下一步
│
├─ 你的主要任务是什么？
│  ├─ 推理 / 模型转换 / 服务部署
│  │  └─ 安装 `ascend-base` + `ascend-inference`
│  ├─ 训练 / 通信 / MindSpeed-LLM
│  │  └─ 安装 `ascend-base` + `ascend-training`
│  ├─ Profiling 采集 / 性能瓶颈分析
│  │  └─ 安装 `ascend-base` + `ascend-profiling`
│  ├─ 算子开发 / Triton 迁移 / op-plugin 接入 / 算子调优
│  │  └─ 安装 `ascend-base` + `ascend-ops`
│  ├─ AI for Science 专项工作
│  │  └─ 安装 `ascend-base` + `ascend-ai-for-science`
│  └─ 我只想要一个非常具体的能力
│     └─ 直接使用 `--skill <skill-name>` 安装单 skill
│
└─ 仍然拿不准？
   └─ 先装 `ascend-base`，再按下面安装指南追加对应 bundle
```

### 推荐路径

1. 新同学先安装 `ascend-base`
2. 再按任务追加 `ascend-inference`、`ascend-training`、`ascend-profiling`、`ascend-ops`
3. 只有在明确知道用途时，再安装更细粒度的 leaf skills

---

## 安装指南

### 推荐安装方式

使用 `npx` 安装到支持 Skills 的 AI 编程工具中。对新同学，推荐优先安装 bundle，而不是一次性安装全部 skills：

```bash
# 基础环境包（推荐所有新同学先装）
npx skills add ascend-ai-coding/awesome-ascend-skills --skill ascend-base

# 推理方向
npx skills add ascend-ai-coding/awesome-ascend-skills --skill ascend-inference

# 训练方向
npx skills add ascend-ai-coding/awesome-ascend-skills --skill ascend-training

# Profiling / 性能分析方向
npx skills add ascend-ai-coding/awesome-ascend-skills --skill ascend-profiling

# 算子开发 / 迁移方向
npx skills add ascend-ai-coding/awesome-ascend-skills --skill ascend-ops

# AI for Science 方向
npx skills add ascend-ai-coding/awesome-ascend-skills --skill ascend-ai-for-science

# 安装单个 Skill
npx skills add ascend-ai-coding/awesome-ascend-skills --skill npu-smi

# 安装全部 Skills（不建议新同学直接使用）
npx skills add ascend-ai-coding/awesome-ascend-skills
```

支持的 AI 编程工具：Claude Code、OpenCode、Cursor、Trae、Codex 等。

### 官方推荐安装包

| 安装包 | 适合谁 | 包含内容 |
|------|------|------|
| `ascend-base` | 所有新同学 | 基础环境、设备检查、容器环境、PyTorch NPU 基础能力 |
| `ascend-inference` | 推理、模型转换、服务部署 | ATC、vLLM-Ascend、ais-bench、量化、Diffusers、Wan 适配 |
| `ascend-training` | 训练、通信、MindSpeed-LLM | HCCL、torch 通信测试、MindSpeed-LLM 全流程、MFU 计算 |
| `ascend-profiling` | Profiling 采集、性能分析 | Profiling 分析、训练 Profiling 采集、算子基准测试、MFU 分析 |
| `ascend-ops` | 算子开发、迁移、调优 | AscendC、op-plugin、Triton-Ascend 迁移、算子基准测试 |
| `ascend-ai-for-science` | AI for Science 专项用户 | AI for Science 总入口及其子技能 |

`ascend-profiling` 和 `ascend-ops` 的区别：

- 选 `ascend-profiling`：你已经有模型/训练任务，重点是**采集 Profiling、定位性能瓶颈、分析 hostbound / computing / communication**。
- 选 `ascend-ops`：你要做的是**算子开发、算子迁移、op-plugin 接入、Triton-Ascend 改写或算子级调优**。

### 领域技能包

如果你已经明确知道自己的方向，也可以直接安装更聚焦的领域技能包：

| 技能包 | 说明 |
|------|------|
| `mindspeed-llm-skills` | MindSpeed-LLM 训练全流程 |
| `diffusers-ascend-skills` | Diffusers 环境、权重准备与推理 |
| `profiling-analysis` | Profiling 分析技能集 |
| `ai-for-science` | AI for Science 技能集 |

### 手动安装

如果无法使用 `npx`，可以手动复制所需的 skill 目录。

**方式一：项目级安装**（推荐）

将所需 skill 复制到项目根目录的 `.agents/skills/` 下：

```bash
# 克隆仓库
git clone https://github.com/ascend-ai-coding/awesome-ascend-skills.git

# 复制基础环境相关 skills 到项目目录
cp -r awesome-ascend-skills/npu-smi your-project/.agents/skills/
cp -r awesome-ascend-skills/ascend-docker your-project/.agents/skills/
cp -r awesome-ascend-skills/torch_npu your-project/.agents/skills/
```

手动安装时，可按上面的“官方推荐安装包”表格选择对应 skill 目录复制。

例如，手动安装 `ascend-ops` 时，至少复制以下目录：

```bash
cp -r awesome-ascend-skills/torch_npu your-project/.agents/skills/
cp -r awesome-ascend-skills/ascendc your-project/.agents/skills/
cp -r awesome-ascend-skills/ascend-opplugin your-project/.agents/skills/
cp -r awesome-ascend-skills/triton-ascend-migration your-project/.agents/skills/
cp -r awesome-ascend-skills/npu-op-benchmark your-project/.agents/skills/
```

**方式二：全局安装**

将 Skill 复制到对应 AI 编程工具的全局 Skills 目录。各平台安装位置请参考官方文档：

| 平台 | 文档链接 |
|------|--------|
| OpenCode | https://opencode.ai/docs/zh-cn/skills/ |
| Cursor | https://cursor.com/cn/docs/context/skills |
| Claude Code | https://code.claude.com/docs/zh-CN/skills |
| Trae | https://docs.trae.cn/ide/skills |


---

## Skill 导航

### 官方推荐入口

先看 bundle，再决定是否深入到单 skill：

| 入口 | 类型 | 说明 |
|------|------|------|
| `ascend-base` | 官方推荐安装包 | 基础环境、设备、容器、PyTorch NPU 基础能力 |
| `ascend-inference` | 官方推荐安装包 | 推理、模型转换、量化、评测、Diffusers、Wan 适配 |
| `ascend-training` | 官方推荐安装包 | 通信测试、MindSpeed-LLM 训练流程、MFU 计算 |
| `ascend-profiling` | 官方推荐安装包 | Profiling 采集、性能分析、算子基准测试 |
| `ascend-ops` | 官方推荐安装包 | AscendC、op-plugin、Triton-Ascend 迁移、算子调优 |
| `ascend-ai-for-science` | 官方推荐安装包 | AI for Science 总入口与子能力 |
| `mindspeed-llm-skills` | 领域技能包 | MindSpeed-LLM 训练全流程 |
| `diffusers-ascend-skills` | 领域技能包 | Diffusers 环境、权重、推理 |
| `profiling-analysis` | 领域技能包 | Profiling 分析技能集 |
| `ai-for-science` | 领域技能包 | AI for Science 技能集 |

### 基础环境与运维

| Skill | 类别 | 描述 |
|------|------|------|
| [npu-smi](npu-smi/SKILL.md) | 运维 | NPU 设备管理：健康状态查询、温度/功耗监控、固件升级、虚拟化配置、证书管理 |
| [ascend-docker](ascend-docker/SKILL.md) | 运维 | Docker 容器配置：NPU 设备映射、卷挂载、开发环境隔离 |
| [torch_npu](torch_npu/SKILL.md) | 开发 | PyTorch 昇腾扩展：环境检查、部署指引、PyTorch 迁移到 NPU 的完整指南 |

### 推理与模型转换

| Skill | 类别 | 描述 |
|------|------|------|
| [atc-model-converter](atc-model-converter/SKILL.md) | 开发 | ATC 模型转换：ONNX 转 .om 格式、OM 推理、精度对比、YOLO 端到端部署 |
| [vllm-ascend](vllm-ascend/SKILL.md) | 开发 | vLLM 推理引擎：离线批推理、OpenAI 兼容 API、量化模型服务、分布式推理 |
| [msmodelslim](msmodelslim/SKILL.md) | 开发 | 模型压缩量化：W4A8/W8A8/W8A8S 量化、MoE/多模态模型支持、精度自动调优 |
| [ais-bench](ais-bench/SKILL.md) | 测试 | AI 模型评估工具：精度评估、性能压测、Function Call |
| [diffusers-ascend-skills](diffusers-ascend/diffusers-ascend-pipeline/SKILL.md) | 技能包 | Diffusers 环境、权重准备与推理 |
| [wan-ascend-adaptation](wan-ascend-adaptation/SKILL.md) | 开发 | Wan 系列视频生成模型及相似扩散框架的昇腾适配指南 |

### 训练与通信

| Skill | 类别 | 描述 |
|------|------|------|
| [hccl-test](hccl-test/SKILL.md) | 测试 | HCCL 集合通信性能测试：带宽测试、AllReduce/AllGather 等基准测试 |
| [torch-npu-comm-test](torch-npu-comm-test/SKILL.md) | 测试 | 通过 torch.distributed 测试通信算子性能，贴近真实训练场景 |
| [mindspeed-llm-skills](mindspeed-llm/mindspeed-llm-pipeline/SKILL.md) | 技能包 | MindSpeed-LLM 环境搭建、数据预处理、权重转换、训练启动 |
| [training-mfu-calculator](training-mfu-calculator/SKILL.md) | 分析 | 大模型训练 MFU 计算、FLOPs 分析与性能报告 |

### Profiling 与性能分析

| Skill | 类别 | 描述 |
|------|------|------|
| [profiling-analysis](profiling-analysis/SKILL.md) | 分析 | Profiling 性能分析技能集：识别下发、通信、计算瓶颈 |
| [mindspeed-llm-train-profiler](mindspeed-llm-train-profiler/SKILL.md) | 分析 | 自动化完成 MindSpeed-LLM 训练 Profiling 数据采集 |
| [npu-op-benchmark](npu-op-benchmark/SKILL.md) | 测试 | 昇腾 NPU 算子性能基准测试 |

### 算子开发与迁移

| Skill | 类别 | 描述 |
|------|------|------|
| [ascendc](ascendc/SKILL.md) | 开发 | AscendC 算子开发：Transformer 算子实现、CANN API 示例 |
| [ascend-opplugin](ascend-opplugin/SKILL.md) | 开发 | op-plugin 环境安装与 torch_npu 自定义算子接入 |
| [triton-ascend-migration](triton-ascend-migration/SKILL.md) | 开发 | GPU/CUDA Triton 算子迁移到 Triton-Ascend |

### 工程知识与专项方向

| Skill | 类别 | 描述 |
|------|------|------|
| [github-issue-summary](github-issue-summary/SKILL.md) | 开发 | 从已关闭 issue 生成故障排查案例、根因分析、经验总结 |
| [github-issue-rca](github-issue-rca/SKILL.md) | 开发 | GitHub Issue 根因分析与调查方向评估 |
| [ai-for-science](ai-for-science/ai4s-main/SKILL.md) | 开发 | AI for Science 总入口：负责 Profiling 采集、模型迁移、路线选择与分流 |

## 外部 Skills (External Skills)

> 以下 skills 从外部仓库自动同步，请勿手动修改。

| Skill | 来源 | 描述 |
|-------|------|------|
| [ascend-inference-repos-copilot](external/gitcode-ascend/ascend-inference-repos-copilot/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 昇腾（Ascend）推理生态开源代码仓库智能问答专家旨在为 vLLM、vLLM-Ascend、MindIE-LLM、MindIE-SD、MindIE-Motor、MindIE-Turbo 以及 ... |
| [ascend-profiling-anomaly](external/gitcode-ascend/ascend-profiling-anomaly/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Analyze Huawei Ascend NPU profiling data to discover hidden performance anomalies and produce a d... |
| [npu-adapter-reviewer](external/gitcode-ascend/npu-adapter-reviewer/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | GPU代码到昇腾NPU适配审查专家。当用户需要将GPU上的代码（特别是深度学习、模型推理相关）迁移到华为昇腾NPU时，必须使用此skill进行全面审查。此skill能识别GPU到NPU迁移的堵点... |
| [simple-vector-triton-gpu-to-npu](external/gitcode-ascend/simple-vector-triton-gpu-to-npu/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 将简单Vector类型Triton算子从GPU迁移到昇腾NPU。当用户需要迁移Triton代码到NPU、提到GPU到NPU迁移、Triton迁移、昇腾适配时使用。注意：无法自动迁移存在编译问题的算子。 |
| [skill-auditor](external/gitcode-ascend/skill-auditor/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Comprehensive security auditor for AI agent skills, prompts, and instructions. Checks for typosqu... |
| [vLLM-ascend_FAQ_Generator](external/gitcode-ascend/vLLM-ascend_FAQ_Generator/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 为 vLLM-ascend 项目构建自动化工作流，处理已关闭的Issue并生成Debug FAQ。Use when users want to process closed issues fro... |
| [vector-triton-ascend-ops-optimizer](external/gitcode-ascend/vector-triton-ascend-ops-optimizer/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 昇腾（Ascend） NPU 上 Triton 算子深度性能优化技能（Skill），致力于实现用户要求的 Triton 算子性能提升。核心技术包括但不限于 Unified Buffer (UB)... |
| [ascend-profiler-db-explorer](external/mindstudio/ascend-profiler-db-explorer/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 面向 Ascend PyTorch Profiler / msprof DB（如 ascend_pytorch_profiler*.db、msprof_*.db）的 SQL 分析技能。将自然语言... |
| [cluster-fast-slow-rank-detector](external/mindstudio/cluster-fast-slow-rank-detector/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 专门用于 Ascend 集群 Profiling 性能数据的“快慢卡”诊断专家技能。当用户提供【集群性能数据目录/路径】并要求分析【快慢卡】、【慢节点】、【负载不均衡】或【集群瓶颈】时，必须触发... |
| [document-ux-review](external/mindstudio/document-ux-review/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 当用户希望你像第一次接触项目的人一样，真实按仓库的 README、安装文档或 quick start 跑一遍，并判断“新人能不能走通”“文档是否可用”“哪里会卡住”“安装/启动说明是否对新手友好... |
| [github-raw-fetch](external/mindstudio/github-raw-fetch/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 当用户提供 GitHub 文件页面链接，或希望读取某个仓库中的源码、配置、README、Markdown、docs 内容时，使用此技能。技能不仅支持将 `github.com/<owner>/<... |
| [mindstudio_profiler_data_check](external/mindstudio/mindstudio_profiler_data_check/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 当用户提供 MindStudio profiler 采集的性能数据（框架 profiler、msprof 命令行）时，对数据完整性、采集状态及关键配置进行校验，确保后续分析工具能正常运行。 |
| [op-mfu-calculator](external/mindstudio/op-mfu-calculator/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 计算算子（如 matmul/GEMM）的 MFU（Machine FLOP Utilization），并给出清晰的公式和推导过程。 |

---

## Skill 工作原理

Skills 采用**渐进式披露**策略管理上下文，确保 AI Agent 高效加载：

1. **发现阶段**：仅加载 `name` + `description`（约 100 tokens）
2. **激活阶段**：触发时加载完整 `SKILL.md` 内容
3. **按需加载**：需要时再加载 `references/` 和 `scripts/` 中的详细文档

这种设计避免了上下文窗口被无关信息占满，同时保证了必要时能获取完整知识。

---

## 治理规范

随着 skills 持续增长，仓库已不再只是平铺的 skill 集合，而是同时包含：官方 bundle、领域技能包、leaf skill、router skill 和 external skills。

当前治理规则见：[`docs/governance/skill-governance.md`](docs/governance/skill-governance.md)

这份规范重点约束：

- taxonomy：skill 属于哪个功能域、扮演什么角色
- naming：何时使用 `ascend-*`、`*-skills`、嵌套 skill 前缀等命名方式
- quality bar：官方 bundle、leaf、router、external 各自的最小质量要求
- analytics / feedback：如何持续发现 bundle 边界问题、重复 skills 和 README 导航问题

如果你在使用过程中遇到“**不知道装哪个**”“**两个 skills 看起来重复**”“**README 导航仍然不清楚**”这类问题，欢迎使用 [`skill feedback` issue 模板](.github/ISSUE_TEMPLATE/skill-feedback.yml) 提反馈。

---

## 贡献指南

欢迎为 Awesome Ascend Skills 贡献新的 Skill 或改进现有内容。

在开始新增 skill 之前，建议先阅读上面的[治理规范](#治理规范)，先判断它应该是：

- 官方 bundle
- 领域技能包
- leaf skill
- router skill
- external synced skill

这样可以减少功能重叠和命名发散。

### 如何编写 SKILL.md

每个 Skill 目录必须包含 `SKILL.md` 文件，遵循以下格式：

```yaml
---
name: skill-name                    # 必须与目录名完全一致
description: 清晰的描述，包含关键词，至少 20 个字符。说明何时使用此 Skill。
keywords:                            # 可选，推荐用于中文/双语 Skill
    - 关键词1
    - 关键词2
---

# Skill 标题

简要介绍...

## 快速开始

简短示例...

## 内容章节

详细说明...

## 官方参考
- [链接标题](url)
```

#### Frontmatter 规则

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | 是 | 必须与目录名完全一致 |
| `description` | 是 | 至少 20 个字符，包含使用场景和关键词 |
| `keywords` | 否 | 推荐添加，用于中文关键词匹配 |

#### 内容规范

- **渐进式披露**：核心内容放在 SKILL.md（不超过 500 行），详细内容放在 `references/`
- **代码块**：始终指定语言（```bash、python、yaml```）
- **表格**：用于结构化参考数据（参数、命令对照）
- **链接**：内部链接使用相对路径，确保可访问

### 目录结构规范

```
skill-name/                          # 目录名：小写 + 连字符
├── SKILL.md                         # 必需：核心内容
├── references/                      # 可选：详细文档
│   ├── installation.md
│   ├── troubleshooting.md
│   └── advanced-usage.md
├── scripts/                         # 可选：可执行脚本
│   ├── check_env.sh
│   └── setup.py
└── assets/                          # 可选：配置文件、模板
    └── config_template.yaml
```

### 命名规范

| 元素 | 规范 | 示例 |
|------|------|------|
| 目录名 | `小写-连字符` | `npu-smi`、`hccl-test` |
| Skill 名 | 匹配目录名 | `name: npu-smi` |
| 脚本文件 | `kebab-case.sh` 或 `snake_case.py` | `npu-health-check.sh` |
| 参考文档 | `小写-连字符.md` | `device-queries.md` |
| 配置文件 | `kebab-case.yaml` | `quant_config_w8a8.yaml` |

### 验证清单

提交前请检查：

- [ ] 已确认该 skill / bundle 的角色类型与功能域
- [ ] `name` 与目录名一致
- [ ] `description` 不少于 20 个字符
- [ ] SKILL.md 有有效的 YAML frontmatter（以 `---` 开始和结束）
- [ ] 内部链接可正常访问
- [ ] 无 `[TODO]` 占位符
- [ ] 已添加到 `.claude-plugin/marketplace.json`
- [ ] 已添加到 README.md 对应导航或安装入口
- [ ] 运行 `python3 scripts/validate_skills.py` 通过

---

## 外部 Skills 同步

本仓库支持自动同步外部 Ascend skills 仓库，实现技能共享和更新。

### 同步机制

同步由 GitHub Actions 自动完成，支持三种触发方式：

1. **定时同步**：每天 UTC 00:00 自动执行
2. **手动触发**：通过 GitHub Actions 页面手动运行
3. **PR 触发**：修改 `.github/external-sources.yml` 配置文件时自动触发

### 添加外部源

编辑 `.github/external-sources.yml` 文件添加新的外部仓库：

```yaml
sources:
  - name: mindstudio                    # 唯一标识，用于 external/{name}/ 目录
    url: https://github.com/kali20gakki/mindstudio-skills
    branch: main                        # 可选，默认 main
    enabled: true                       # 可选，默认 true
```

### 同步规则

- **存储位置**：`external/{source-name}/{skill-name}/`
- **冲突策略**：同名 skill 以本仓为准，外部 skill 被跳过
- **来源标记**：同步的 skill 会自动添加 `synced-from`、`synced-date`、`synced-commit` 等属性
- **PR 审核**：同步结果生成 PR，需人工审核后合并

### 查看外部 Skills

已同步的外部 skills 会显示在本 README 的"外部 Skills"表格中。

---

## 提交 PR

### 准备工作

1. **Fork 仓库**：点击 GitHub 页面右上角的 Fork 按钮
2. **克隆 Fork**：
   ```bash
   git clone https://github.com/YOUR_USERNAME/awesome-ascend-skills.git
   cd awesome-ascend-skills
   ```
3. **创建分支**：
   ```bash
   git checkout -b feat/your-skill-name
   # 或
   git checkout -b fix/description-of-fix
   ```

### 开发流程

1. **创建 Skill 目录**：
   ```bash
   mkdir -p your-skill-name
   cp skills/template/SKILL.md your-skill-name/
   ```

2. **编写 SKILL.md**：按照[贡献指南](#贡献指南)的规范编写

3. **本地验证**：
   ```bash
   python3 scripts/validate_skills.py
   ```

4. **更新注册表**：在 `.claude-plugin/marketplace.json` 中添加新 Skill 条目

5. **更新 README.md**：在 Skill 列表表格中添加新行

### 提交规范

- **Commit 信息**：使用清晰的描述，例如：
  - `feat: add npu-smi skill`
  - `fix: update msmodelslim quantization params`
  - `docs: improve hccl-test examples`

### PR 模板

提交 PR 时，请参考：```./guideline/.github/PULL_REQUEST_TEMPLATE.md```

### 审核流程

1. 维护者会在 3 个工作日内审核
2. 根据反馈进行修改
3. 审核通过后合并到 main 分支

---

## 官方文档

- [华为昇腾官方文档](https://www.hiascend.com/document)
- [npu-smi 命令参考](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/envdeployment/instg/instg_0045.html)
- [CANN 开发指南](https://www.hiascend.com/document/detail/zh/canncommercial/)

---

## 许可证

MIT License

Copyright (c) 2024 Ascend AI Coding

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
