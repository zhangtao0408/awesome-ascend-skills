# Skill Governance

本文件定义 Awesome Ascend Skills 的 Phase 3 治理规则，用于解决 skills 持续增长后的分类、命名、整合与质量控制问题。

## 1. 治理目标

治理的目标不是增加流程，而是让仓库在持续扩张时仍然保持：

1. **可发现**：新人能快速找到合适入口
2. **可维护**：贡献者知道应该新增什么类型的 skill，而不是随意增加平行能力
3. **可演进**：后续可以基于统一 taxonomy 做整合、拆分、下线和 bundle 调整
4. **可衡量**：可以围绕 bundle adoption、skill 选择效率、README 导航效率收集反馈

## 2. Skill Taxonomy

仓库中的 skill 统一分为四类角色。

### 2.1 Official Bundle

面向用户直接安装的官方推荐入口，解决“应该先装什么”的问题。

- 命名规则：`ascend-<domain>`
- 例子：`ascend-base`、`ascend-inference`、`ascend-training`、`ascend-profiling`、`ascend-ops`
- 设计原则：
  - 面向任务场景，而不是面向实现细节
  - 默认优先推荐给新用户
  - 一个 bundle 应该有清晰边界，避免“什么都放进去”

### 2.2 Domain Skill Set

聚焦某个技术领域的技能包，适合已经明确方向的用户。

- 常见命名：`<domain>-skills`
- 例子：`mindspeed-llm-skills`、`diffusers-ascend-skills`
- 设计原则：
  - 聚焦单一领域
  - 可被官方 bundle 引用
  - 不要求承担新手入口职责

### 2.3 Leaf Skill / Router Skill

最细粒度的技能单元，或承担任务分流职责的入口 skill。

- leaf skill：解决单一问题，如 `npu-smi`、`hccl-test`
- router skill：负责识别任务并分流到子 skill，如 `ai-for-science/ai4s-main`
- 设计原则：
  - leaf skill 的目标必须明确且边界清晰
  - router skill 必须说明“何时分流、分流到哪里、不会做什么”

### 2.4 External Synced Skill

从外部仓库自动同步的 skills。

- 命名规则：`external-<source>-<skill>`
- 存储路径：`external/{source-name}/{skill-name}/`
- 设计原则：
  - 不作为默认新手入口
  - 需要与本仓技能明确边界
  - 不直接修改同步内容，改动通过外部源或同步流程完成

## 3. Functional Domains

除角色外，所有 skills 还应归入一个功能域，便于 README 导航和后续统计。

| Domain | 说明 | 当前代表能力 |
|------|------|------|
| `base` | 基础环境与运维 | `npu-smi`、`ascend-docker`、`torch_npu` |
| `inference` | 推理、模型转换、量化、评测 | `atc-model-converter`、`vllm-ascend`、`ais-bench`、`msmodelslim` |
| `training` | 训练、通信、训练链路 | `hccl-test`、`torch-npu-comm-test`、`mindspeed-llm-*` |
| `profiling` | Profiling 采集与性能分析 | `profiling-analysis`、`mindspeed-llm-train-profiler` |
| `ops` | 算子开发、迁移与调优 | `ascendc`、`ascend-opplugin`、`triton-ascend-migration` |
| `ai-for-science` | AI for Science 专项方向 | `ai-for-science/*` |
| `knowledge` | 工程案例、排障知识沉淀 | `github-issue-summary`、`github-issue-rca` |

新增 skill 时，应先判断功能域，再决定角色类型。

### 3.0 开发入口目录

为降低仓库开发时的查找成本，仓库应提供按功能域组织的开发入口目录：

- `base/`
- `inference/`
- `training/`
- `profiling/`
- `ops/`
- `knowledge/`
- `ai-for-science/`

当前仓库已采用分类目录承载本地 skills，并要求：

1. README 中有明确的开发目录入口
2. 每个分类目录与其承载的实际 skill 保持一致
3. validate / CI / marketplace / sync 必须与当前分类目录结构一致

### 3.1 `profiling` vs `ops` 边界

这两个域最容易让新人混淆，需要显式区分：

- `profiling`：关注**发现瓶颈**，核心问题是“哪里慢、慢在哪里、该怎么分析”。
- `ops`：关注**开发和改造算子本身**，核心问题是“怎么实现/迁移/接入/优化一个算子”。

允许存在少量交叉工具（例如算子 benchmark 既能辅助 profiling，也能辅助 ops 调优），但在 README 和 bundle 描述中必须明确主职责，避免两个 bundle 看起来像同一件事。

## 4. 命名与收敛规则

### 4.1 命名规则

- 官方 bundle：`ascend-<domain>`
- 领域 skill set：优先使用 `-skills` 后缀
- 分类入口目录：优先使用功能域名，如 `base/`、`inference/`、`ops/`
- 根目录 leaf skill：保持 `lowercase-with-hyphens`
- 分类目录下的 leaf skill：仍使用原始 leaf 名称，不额外追加分类前缀
- 分类目录下的 nested skill：名称以前一层领域子目录为前缀，不额外追加分类目录前缀
- 嵌套 skill：遵循仓库校验器要求，名称以前缀目录名开头
- external skill：保持同步命名，不在本仓重命名

### 4.2 何时新建 bundle

只有同时满足以下条件，才建议新增官方 bundle：

1. 面向一个稳定、可识别的用户场景
2. 至少包含 3 个以上经常一起使用的能力
3. 能显著降低安装与选择成本
4. 与现有 bundle 不形成明显重叠

### 4.3 何时新增 leaf skill

适合新增 leaf skill 的情况：

- 解决一个明确、独立的问题
- 无法自然纳入已有 leaf skill
- 不应仅仅因为文档过长就新建 skill；优先考虑放入 `references/`

### 4.4 何时整合而不是新增

遇到以下情况，优先整合：

- 两个 skills 的触发场景高度相似
- 两个 skills 只是在参数、模型或数据来源上略有差异
- 用户视角下它们仍属于同一任务路径

## 5. Quality Bar

### 5.1 All Skills 必须满足

- `SKILL.md` 具备合法 frontmatter
- `name` 与目录/嵌套命名规则一致
- `description` 至少 20 字，能明确说明触发场景
- 包含 `Quick Start` 或等价快速示例
- 无 `[TODO]` 占位符
- 相对链接可访问
- 通过 `python3 scripts/validate_skills.py`（外部历史问题需单独说明）

### 5.2 Official Bundle 额外要求

- 必须在 README 中有明确安装示例
- 必须在 README 的 decision tree / Quick Start 中有清晰入口
- 必须在 README 中说明适合谁使用
- 必须列出核心覆盖范围，避免“万能包”描述
- bundle 中每个 skill 的存在都应能回答“为什么用户会一起安装它”
- 如果 bundle 与其他 bundle 容易混淆，必须补充“怎么选”的边界说明

### 5.5 开发入口目录要求

- 每个分类入口目录都应有 `README.md`
- 分类入口目录中的说明必须指向当前实际 skill 路径
- 如果目录结构发生迁移，README、marketplace、CI 与交叉链接必须在同一轮更新中完成

### 5.3 Router Skill 额外要求

- 明确写出路由条件
- 明确列出会分流到哪些子 skill
- 明确列出不适用范围，避免过度触发

### 5.4 External Skill 额外要求

- 在 README 中标识来源
- 默认不进入官方推荐安装路径
- 如与本仓 skill 重叠，应优先说明边界而不是简单并列

## 6. Contribution Decision Flow

在新增 skill 前，按以下顺序判断：

1. 这是已有 leaf skill 的补充，还是新的任务？
2. 这个需求应该放进现有 `references/`，还是值得成为独立 skill？
3. 它属于哪个功能域？
4. 它是 leaf、router、domain skill set，还是需要升级为官方 bundle？
5. README / marketplace / 导航是否需要同步更新？

## 7. Minimal Analytics / Feedback Loop

当前阶段先采用“轻量治理”，不引入复杂埋点系统，而是通过以下方式收集信号：

1. 使用 GitHub issue 模板收集：
   - 新人不知道装什么
   - 某个 bundle 太大/太小
   - 两个 skills 重叠
   - README 导航仍然不清楚
2. 在每月或每个 release 周期复盘：
   - 新增了多少 leaf skills
   - 新增了多少 bundle
   - 是否出现高重叠技能
   - 哪些 bundle 最常被提及或反馈

推荐跟踪的指标：

- `bundle_adoption_feedback_count`
- `skill_selection_confusion_count`
- `readme_navigation_feedback_count`
- `duplicate_skill_candidates`

## 8. Governance Checklist

提交涉及 skill 新增/重构的 PR 时，建议自查：

- [ ] 已明确该变更属于哪个功能域
- [ ] 已明确该变更是 bundle / domain skill set / leaf / router / external 哪一类
- [ ] 未与现有 skill 产生明显重复
- [ ] README 与 marketplace 已同步更新
- [ ] 如为官方 bundle，已说明推荐用户与核心覆盖范围
- [ ] 如为 router，已明确分流规则
- [ ] 如为 external，已保持来源边界清晰

这份文档会随着仓库演进继续更新，但在当前阶段，以上规则应作为默认准则执行。
