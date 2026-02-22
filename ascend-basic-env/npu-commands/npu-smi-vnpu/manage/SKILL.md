---
name: npu-smi-vnpu-manage
description: npu-smi vNPU management commands for Huawei Ascend NPU. Use when creating, destroying, or configuring vNPU instances.
---

# npu-smi vNPU Management

Manage virtual NPU instances using `npu-smi set`.

## Quick Reference

```bash
npu-smi set -t vnpu-mode -d 0                             # Set AVI mode
npu-smi set -t create-vnpu -i 0 -c 0 -f template -v 103   # Create vNPU
npu-smi set -t destroy-vnpu -i 0 -c 0 -v 103              # Destroy vNPU
```

## Commands

### Set AVI Mode

```bash
npu-smi set -t vnpu-mode -d <mode>
```

**Modes:**
| Value | Mode |
|-------|------|
| 0 | Container |
| 1 | VM |

### Create vNPU

```bash
npu-smi set -t create-vnpu -i <id> -c <chip_id> -f <template> [-v <vnpu_id>] [-g <vgroup_id>]
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| template | Template name |
| vnpu_id | Optional: [phy_id*16+100, phy_id*16+115] |
| vgroup_id | Optional: Group ID [0,1,2,3] |

### Destroy vNPU

```bash
npu-smi set -t destroy-vnpu -i <id> -c <chip_id> -v <vnpu_id>
```

## Examples

### Create and Destroy vNPU

```bash
#!/bin/bash

NPU=0
CHIP=0

echo "Creating vNPU..."
npu-smi set -t create-vnpu -i $NPU -c $CHIP -f vir02 -v 103

echo "Verifying..."
npu-smi info -t info-vnpu -i $NPU -c $CHIP

# Destroy when done
# npu-smi set -t destroy-vnpu -i $NPU -c $CHIP -v 103
```

## Related Skills

- [../query/](query/SKILL.md) - Query vNPU information
- [../../npu-smi-info/basic/](npu-smi-info/basic/SKILL.md) - Device info
