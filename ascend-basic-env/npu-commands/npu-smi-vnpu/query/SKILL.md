---
name: npu-smi-vnpu-query
description: npu-smi vNPU query commands for Huawei Ascend NPU. Use when checking AVI mode, listing templates, or viewing vNPU information.
---

# npu-smi vNPU Queries

Query virtualization information using `npu-smi info`.

## Quick Reference

```bash
npu-smi info -t vnpu-mode            # Query AVI mode
npu-smi info -t template-info        # List templates
npu-smi info -t info-vnpu -i 0 -c 0  # View vNPU info
```

## Commands

### Query AVI Mode

```bash
npu-smi info -t vnpu-mode
```

**Output:**
| Value | Mode |
|-------|------|
| 0 | Container |
| 1 | VM |

### Query Templates

```bash
npu-smi info -t template-info
npu-smi info -t template-info -i <id>
```

### Query vNPU Info

```bash
npu-smi info -t info-vnpu -i <id> -c <chip_id>
```

**Output:**
- vNPU ID
- vNPU Group ID
- AI Core Num
- Memory Size
- Status

## Examples

### List vNPU Status

```bash
#!/bin/bash

NPU=0
CHIP=0

echo "AVI Mode:"
npu-smi info -t vnpu-mode

echo ""
echo "vNPU Info:"
npu-smi info -t info-vnpu -i $NPU -c $CHIP
```

## Related Skills

- [../manage/](manage/SKILL.md) - Create and destroy vNPU
- [../../npu-smi-info/basic/](npu-smi-info/basic/SKILL.md) - Device information
