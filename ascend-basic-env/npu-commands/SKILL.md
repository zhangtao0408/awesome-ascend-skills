---
name: npu-commands
description: Ascend NPU command-line utilities and hardware management master skill. Use for npu-smi usage, device management, monitoring, and basic hardware operations. Routes to specialized sub-skills organized by function: queries (basic/advanced), configuration (thresholds/modes/fan/system/clear), upgrades (workflow/components), virtualization (query/manage), and certificates (query/manage/monitor).
---

# npu-smi Command Reference

Master skill for Huawei Ascend NPU command-line utilities. This skill organizes all npu-smi functionality into focused sub-skills by function.

## Overview

The `npu-smi` utility provides comprehensive management capabilities for Ascend NPU devices. To keep skills lightweight and focused, functionality is split into **14 specialized sub-skills** organized by category.

## Quick Start

```bash
# List devices
npu-smi info -l

# Check health
npu-smi info -t health -i 0

# View device details
npu-smi info -t board -i 0
```

## Sub-Skill Organization

### 📊 1. Device Queries (npu-smi-info/)

Query device status and information.

| Sub-Skill | Use When | Key Commands |
|-----------|----------|--------------|
| [npu-smi-info/basic/](npu-smi-info/basic/SKILL.md) | Listing devices, checking health, viewing board/chip info, temperature, power, memory | `info -l`, `info -t health`, `info -t board`, `info -t temp`, `info -t power`, `info -t memory` |
| [npu-smi-info/advanced/](npu-smi-info/advanced/SKILL.md) | Checking processes, ECC errors, utilization, PCIe | `info proc`, `info -t ecc`, `info -t usages`, `info -t pcie-info` |

**Navigation Guide:**
- Use **basic** for device discovery, health checks, and real-time metrics (temperature, power, memory)
- Use **advanced** for troubleshooting and detailed diagnostics

### ⚙️ 2. Configuration (npu-smi-config/)

Configure device settings and parameters.

| Sub-Skill | Use When | Key Features |
|-----------|----------|--------------|
| [npu-smi-config/thresholds/](npu-smi-config/thresholds/SKILL.md) | Setting temperature and power limits | Temperature thresholds, power limits |
| [npu-smi-config/modes/](npu-smi-config/modes/SKILL.md) | Configuring ECC, compute modes, P2P | ECC mode, compute mode, persistence, P2P |
| [npu-smi-config/fan/](npu-smi-config/fan/SKILL.md) | Controlling fan speed and mode | Manual/auto mode, speed ratio |
| [npu-smi-config/system/](npu-smi-config/system/SKILL.md) | System-level settings | MAC address, boot medium, CPU freq, logging |
| [npu-smi-config/clear/](npu-smi-config/clear/SKILL.md) | Clearing counters and errors | ECC error clearing, cert threshold reset |

**Navigation Guide:**
- Use **thresholds** to set safety limits for temperature and power
- Use **modes** to configure operational modes (ECC, compute, etc.)
- Use **fan** for thermal management and noise control
- Use **system** for network, boot, and logging configuration
- Use **clear** to reset error counters after issues resolved

### 🔧 3. Firmware Upgrades (npu-smi-upgrade/)

Manage firmware lifecycle.

| Sub-Skill | Use When | Key Features |
|-----------|----------|--------------|
| [npu-smi-upgrade/workflow/](npu-smi-upgrade/workflow/SKILL.md) | Understanding upgrade process | Complete workflow: query → upgrade → activate |
| [npu-smi-upgrade/components/](npu-smi-upgrade/components/SKILL.md) | Upgrading specific components | MCU, bootloader, VRD specifics |

**Navigation Guide:**
- Use **workflow** for the complete upgrade procedure
- Use **components** for component-specific details and requirements

**Important Notes:**
- MCU: Requires restart after activation
- Bootloader: Requires restart after activation
- VRD: Requires power cycle (30+ seconds off)

### 🖥️ 4. Virtualization (npu-smi-vnpu/)

Manage virtual NPU instances.

| Sub-Skill | Use When | Key Features |
|-----------|----------|--------------|
| [npu-smi-vnpu/query/](npu-smi-vnpu/query/SKILL.md) | Checking virtualization status | AVI mode, templates, vNPU info |
| [npu-smi-vnpu/manage/](npu-smi-vnpu/manage/SKILL.md) | Creating and managing vNPU | Create, destroy, configure vNPU |

**Navigation Guide:**
- Use **query** to check current virtualization state
- Use **manage** to create, destroy, or modify vNPU instances

**Common Use Cases:**
- Multi-tenant environments
- Resource isolation
- Development/testing isolation

### 🔐 5. Certificates (npu-smi-cert/)

Manage TLS certificates and security.

| Sub-Skill | Use When | Key Features |
|-----------|----------|--------------|
| [npu-smi-cert/query/](npu-smi-cert/query/SKILL.md) | Viewing certificate information | CSR generation, cert details, rootkey |
| [npu-smi-cert/manage/](npu-smi-cert/manage/SKILL.md) | Importing and configuring certs | Import TLS, set thresholds, restore defaults |
| [npu-smi-cert/monitor/](npu-smi-cert/monitor/SKILL.md) | Monitoring certificate expiration | Expiration checks, alerting scripts |

**Navigation Guide:**
- Use **query** to view current certificates and generate CSR
- Use **manage** to import new certificates and configure settings
- Use **monitor** to track expiration and set up alerts

**Security Best Practices:**
- Regular certificate rotation
- Monitor expiration dates
- Use appropriate thresholds (30 days prod, 60 days staging)

## Decision Tree

```
Need to check device status?
├── Basic info / Temp / Power / Memory → npu-smi-info/basic/
├── Processes/Errors/Utilization → npu-smi-info/advanced/

Need to change settings?
├── Temperature/Power limits → npu-smi-config/thresholds/
├── ECC/Compute/P2P modes → npu-smi-config/modes/
├── Fan control → npu-smi-config/fan/
├── MAC/Boot/Logging → npu-smi-config/system/
├── Clear errors → npu-smi-config/clear/

Upgrading firmware?
├── Learn workflow → npu-smi-upgrade/workflow/
├── Component details → npu-smi-upgrade/components/

Working with virtualization?
├── Check status → npu-smi-vnpu/query/
├── Create/Destroy vNPU → npu-smi-vnpu/manage/

Managing certificates?
├── View/Generate CSR → npu-smi-cert/query/
├── Import/Configure → npu-smi-cert/manage/
├── Monitor expiration → npu-smi-cert/monitor/
```

## Prerequisites

- npu-smi tool installed
- Root permissions (configuration/upgrade)
- Runtime permissions (query operations)

## Parameter Reference

| Parameter | Description | How to Get |
|-----------|-------------|------------|
| `id` | Device ID | `npu-smi info -l` |
| `chip_id` | Chip ID | `npu-smi info -m` |
| `vnpu_id` | vNPU ID | Auto-assigned or specified |

## Supported Platforms

- Atlas 200I DK A2 Developer Kit
- Atlas 500 A2 Smart Station
- Atlas 200I A2 Acceleration Module (RC/EP scenarios)

## Important Notes

- Most configuration commands require **root permissions**
- Device IDs from `npu-smi info -l`
- Chip IDs from `npu-smi info -m`
- Command availability varies by hardware platform
- MAC/boot changes require restart
- MCU/bootloader require restart after activation
- VRD requires power cycle (30+ seconds)

## Official Documentation

- **npu-smi Reference**: https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/envdeployment/instg/instg_0045.html
