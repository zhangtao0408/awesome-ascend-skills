---
name: npu-smi-upgrade-components
description: npu-smi firmware component upgrades for Huawei Ascend NPU. Use when upgrading MCU, bootloader, or VRD firmware components.
---

# npu-smi Component Upgrades

Upgrade specific firmware components.

## Components

| Component | Restart Required |
|-----------|------------------|
| MCU | Yes |
| Bootloader | Yes |
| VRD | Power cycle (30s) |

## MCU Upgrade

```bash
npu-smi upgrade -t mcu -i 0 -f mcu.hpm
npu-smi upgrade -a mcu -i 0
# Restart required
```

## Bootloader Upgrade

```bash
npu-smi upgrade -t bootloader -i 0 -f bootloader.hpm
npu-smi upgrade -a bootloader -i 0
# Restart required
```

## VRD Upgrade

```bash
npu-smi upgrade -t vrd -i 0
npu-smi upgrade -a vrd -i 0
# Power cycle required (30+ seconds off)
```

## Related Skills

- [../workflow/](workflow/SKILL.md) - Complete workflow
- [../../npu-smi-config/system/](npu-smi-config/system/SKILL.md) - Power state control
