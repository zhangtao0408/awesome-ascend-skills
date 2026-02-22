---
name: npu-smi-config-system
description: npu-smi system configuration for Huawei Ascend NPU. Use when setting MAC addresses, boot medium, CPU frequency, and system logging.
---

# npu-smi System Configuration

Configure system settings.

## Quick Reference

```bash
npu-smi set -t mac-addr -i 0 -c 0 -d 0 -s "XX:XX:XX:XX:XX:XX"
npu-smi set -t boot-select -i 0 -c 0 -d 3
npu-smi set -t cpu-freq-up -i 0 -d 0
npu-smi set -t sys-log-enable -d 1
```

## Commands

### MAC Address

- mac_id: 0=eth0, 1=eth1, 2=eth2, 3=eth3
- mac_string: Format "XX:XX:XX:XX:XX:XX"

### Boot Medium

| Value | Medium |
|-------|--------|
| 3 | M.2 SSD |
| 4 | eMMC |

### CPU Frequency

| Value | CPU | AI Core |
|-------|-----|---------|
| 0 | 1.9 GHz | 800 MHz |
| 1 | 1.0 GHz | 800 MHz |

### System Logging

| Value | Mode |
|-------|------|
| 0 | Disable |
| 1 | Enable |

## Related Skills

- [../../npu-smi-upgrade/workflow/](npu-smi-upgrade/workflow/SKILL.md) - Firmware upgrade
