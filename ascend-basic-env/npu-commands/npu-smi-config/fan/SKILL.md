---
name: npu-smi-config-fan
description: npu-smi fan control for Huawei Ascend NPU. Use when configuring fan modes and speed.
---

# npu-smi Fan Control

Configure fan settings.

## Quick Reference

```bash
npu-smi set -t pwm-mode -d 1
npu-smi set -t pwm-mode -d 0
npu-smi set -t pwm-duty-ratio -d 50
```

## Commands

### Set Fan Mode

| Value | Mode |
|-------|------|
| 0 | Manual |
| 1 | Automatic |

### Set Fan Speed

Range: [0-100] percent

## Examples

### Temperature-Based Control

```bash
TEMP=$(npu-smi info -t temp -i 0 -c 0 | grep "NPU" | awk '{print $4}')
npu-smi set -t pwm-mode -d 0
npu-smi set -t pwm-duty-ratio -d 80
```

## Related Skills

- [../thresholds/](thresholds/SKILL.md) - Temperature thresholds
- [../../npu-smi-info/metrics/](npu-smi-info/metrics/SKILL.md) - Query temperature
