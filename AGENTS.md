# AGENTS.md - Guide for AI Coding Agents

Coding guidelines for the Awesome Ascend Skills repository.

## Repository Overview

A **knowledge base repository** for Huawei Ascend NPU development, structured as AI Skills with multiple layers:

- root-level leaf skills
- nested domain skills
- official bundle entries in `marketplace.json`
- external synced skills under `external/`

**Primary Skills:** npu-smi, hccl-test, atc-model-converter, ascend-docker, msmodelslim, ais-bench, vllm-ascend, ascendc, torch_npu

---

## Build/Test/Validate Commands

```bash
# Validate all skills
python3 scripts/validate_skills.py

# Validate single skill
python3 scripts/validate_skills.py | grep -A 3 "your-skill/"

# Quick checks
head -20 base/npu-smi/SKILL.md
grep "^name:" base/npu-smi/SKILL.md
```

**CI/CD:** `.github/workflows/validate-skills.yml` runs on push to `main`/`feat/**` and PRs.

---

## Repository Structure

```
awesome-ascend-skills/
├── base/                              # Category entry + categorized leaf skills
│   ├── README.md
│   ├── npu-smi/
│   ├── ascend-docker/
│   └── torch_npu/
├── inference/                         # Inference/model conversion/benchmark skills
│   ├── README.md
│   ├── atc-model-converter/
│   ├── msmodelslim/
│   └── diffusers-ascend/
├── training/                          # Training/communication skill trees
│   ├── README.md
│   ├── hccl-test/
│   ├── torch-npu-comm-test/
│   └── mindspeed-llm/
├── profiling/                         # Profiling/analysis skill trees
│   ├── README.md
│   ├── profiling-analysis/
│   └── mindspeed-llm-train-profiler/
├── ops/                               # Operator development/migration skills
│   ├── README.md
│   ├── ascendc/
│   ├── ascend-opplugin/
│   └── triton-ascend-migration/
├── knowledge/                         # Engineering knowledge and RCA skills
│   ├── README.md
│   ├── github-issue-summary/
│   └── github-issue-rca/
├── ai-for-science/                    # Router + nested specialist skills
├── external/                          # Synced external skills
├── docs/governance/                   # Repository governance docs
├── scripts/validate_skills.py         # Validation script
└── .claude-plugin/marketplace.json    # Plugin registry and bundle definitions
```

---

## Code Style Guidelines

### SKILL.md Format

```yaml
---
name: skill-name                    # MUST match directory name
description: Clear description (≥20 chars)
keywords:                            # Optional
    - keyword1
---

# Skill Title

## Quick Start
Brief examples...

## Content Sections
Detailed instructions...
```

**Rules:**
- Root-level `name`: MUST match directory name exactly
- Categorized leaf `name`: when stored under `base/`, `inference/`, `training/`, `profiling/`, `ops/`, or `knowledge/`, still matches the leaf directory name exactly
- Categorized nested `name`: when a nested skill tree lives under a category directory, the nested leaf still follows the domain subfolder prefix (for example `training/mindspeed-llm/... -> name: mindspeed-llm-*`)
- Nested `name`: MUST follow validator rules and start with the top-level folder prefix (for example `ai-for-science-*`)
- `description`: ≥20 characters for agent matching
- **Progressive disclosure:** Core in SKILL.md (≤500 lines), details in `references/`
- **Bilingual:** Chinese content encouraged
- **Code blocks:** Always specify language (```bash, python```)
- **Links:** Use relative paths

### Python Scripts
```python
#!/usr/bin/env python3
from typing import Dict, List, Tuple

def validate(path: str) -> Tuple[List[str], List[str]]:
    """Docstring."""
    return [], []
```

### Shell Scripts
```bash
#!/bin/bash
set -e
readonly DIR="$(cd "$(dirname "$0")" && pwd)"
```

---

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Directories | `lowercase-with-hyphens` | `npu-smi` |
| Root skill names | Match directory | `name: npu-smi` |
| Categorized leaf names | Match leaf directory | `knowledge/github-issue-rca -> name: github-issue-rca` |
| Nested skill names | Start with top-level folder | `name: ai-for-science-ankh-...` |
| Official bundles | `ascend-<domain>` | `ascend-inference` |
| Domain skill sets | Prefer `-skills` suffix | `mindspeed-llm-skills` |
| Scripts | `kebab-case.sh` / `snake_case.py` | `check-health.sh` |
| References | `lowercase-with-hyphens.md` | `queries.md` |
| Configs | `kebab-case.yaml` | `config.yaml` |

---

## Error Handling

**Python:** Explicit errors, actionable messages, `sys.exit(1)` on failure.

**Shell:** Use `set -e`, check return codes, provide fallback.

---

## Adding New Skills

1. `mkdir -p new-skill`
2. Create `SKILL.md` with frontmatter
3. Add `references/`, `scripts/`, `assets/` as needed
4. Decide whether this is a leaf skill, nested skill, domain skill set, official bundle, or external sync concern
5. Update `.claude-plugin/marketplace.json` if registry or bundle exposure changes
6. Update `README.md` navigation / install entry, not just a flat table
7. If governance rules are affected, update `docs/governance/skill-governance.md`
8. Run `python3 scripts/validate_skills.py`

**marketplace.json:**
```json
{
  "name": "new-skill",
  "description": "Description for agent matching",
  "source": "./new-skill",
  "category": "development"
}
```

---

## Key Principles

1. **Layered structure:** root skills, nested skills, bundles, and external skills each have distinct roles
2. **Independence:** Each skill usable independently
3. **Keywords:** Include in `description`
4. **Progressive disclosure:** Core in SKILL.md, details in `references/`
5. **Bilingual:** Chinese and English acceptable

---

## Validation Checklist

Before submitting PR:
- [ ] `name` matches directory name
- [ ] Nested skills follow top-level prefix naming rules
- [ ] `description` ≥20 characters
- [ ] Valid YAML frontmatter
- [ ] Internal links resolve
- [ ] No `[TODO]` placeholders
- [ ] Added to `marketplace.json` and appropriate README navigation / install entry
- [ ] `python3 scripts/validate_skills.py` passes

---

## References

- Huawei Ascend: https://www.hiascend.com/document
