---
name: npu-smi-config-clear
description: npu-smi clear commands for Huawei Ascend NPU. Use when clearing ECC errors and resetting certificate thresholds.
---

# npu-smi Clear Commands

Clear counters and settings.

## Quick Reference

```bash
npu-smi clear -t ecc-info -i 0 -c 0
npu-smi clear -t tls-cert-period -i 0 -c 0
```

## Commands

### Clear ECC Errors

```bash
npu-smi clear -t ecc-info -i <id> -c <chip_id>
```

### Restore Certificate Threshold

```bash
npu-smi clear -t tls-cert-period -i <id> -c <chip_id>
```

## Related Skills

- [../modes/](modes/SKILL.md) - ECC mode configuration
- [../../npu-smi-info/advanced/](npu-smi-info/advanced/SKILL.md) - Check ECC status
