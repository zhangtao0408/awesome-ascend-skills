---
name: npu-smi-config-modes
description: npu-smi mode configuration for Huawei Ascend NPU. Use when configuring ECC mode, compute mode, persistence mode, and P2P.
---

# npu-smi Mode Configuration

Configure operational modes.

## Quick Reference

```bash
npu-smi set -t ecc-mode -i 0 -c 0 -d 1
npu-smi set -t compute-mode -i 0 -c 0 -d 0
npu-smi set -t persistence-mode -i 0 -d 1
npu-smi set -t p2p-mem-cfg -i 0 -c 0 -d 1
```

## Commands

### ECC Mode

| Value | Mode |
|-------|------|
| 0 | Disable |
| 1 | Enable |

### Compute Mode

| Value | Mode |
|-------|------|
| 0 | Default |
| 1 | Exclusive |
| 2 | Prohibited |

### Persistence Mode

| Value | Mode |
|-------|------|
| 0 | Disable |
| 1 | Enable |

### P2P Configuration

| Value | Mode |
|-------|------|
| 0 | Disable |
| 1 | Enable |

## Related Skills

- [../clear/](clear/SKILL.md) - Clear ECC errors
- [../../npu-smi-info/advanced/](npu-smi-info/advanced/SKILL.md) - Check ECC status
