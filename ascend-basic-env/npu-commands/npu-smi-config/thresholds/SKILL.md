---
name: npu-smi-config-thresholds
description: npu-smi threshold configuration for Huawei Ascend NPU. Use when setting temperature thresholds and power limits.
---

# npu-smi Threshold Configuration

Configure temperature and power thresholds.

## Quick Reference

```bash
npu-smi set -t temperature -i 0 -c 0 -d 85
npu-smi set -t power-limit -i 0 -c 0 -d 300
```

## Commands

### Set Temperature Threshold

```bash
npu-smi set -t temperature -i <id> -c <chip_id> -d <value>
```

**Parameters:**
- value: Temperature threshold in °C

### Set Power Limit

```bash
npu-smi set -t power-limit -i <id> -c <chip_id> -d <value>
```

**Parameters:**
- value: Power limit in Watts

## Examples

### Configure Safe Defaults

```bash
npu-smi set -t temperature -i 0 -c 0 -d 80
npu-smi set -t power-limit -i 0 -c 0 -d 310
```

## Related Skills

- [../../npu-smi-info/metrics/](npu-smi-info/metrics/SKILL.md) - Query current metrics
- [../modes/](modes/SKILL.md) - Configure modes
