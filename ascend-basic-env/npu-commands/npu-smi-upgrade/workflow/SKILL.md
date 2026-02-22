---
name: npu-smi-upgrade-workflow
description: npu-smi firmware upgrade workflow for Huawei Ascend NPU. Use when performing complete firmware upgrades including version queries and activation.
---

# npu-smi Upgrade Workflow

Complete firmware upgrade process.

## Quick Reference

```bash
npu-smi upgrade -b mcu -i 0
npu-smi upgrade -t mcu -i 0 -f file.hpm
npu-smi upgrade -q mcu -i 0
npu-smi upgrade -a mcu -i 0
```

## Workflow

```
Query → Upgrade → Check Status → Activate → Restart
```

## Commands

### Query Version

```bash
npu-smi upgrade -b <item> -i <id>
```

### Upgrade

```bash
npu-smi upgrade -t <item> -i <id> -f <file>
```

### Check Status

```bash
npu-smi upgrade -q <item> -i <id>
```

### Activate

```bash
npu-smi upgrade -a <item> -i <id>
```

## Related Skills

- [../components/](components/SKILL.md) - Component-specific details
