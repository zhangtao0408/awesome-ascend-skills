---
name: npu-smi-info-advanced
description: npu-smi advanced info queries for Huawei Ascend NPU. Use when checking processes, ECC errors, utilization, PCIe, P2P, and product information.
---

# npu-smi Advanced Queries

Query advanced device information using `npu-smi info`.

## Quick Reference

```bash
npu-smi info proc -i 0 -c 0        # Running processes
npu-smi info -t ecc -i 0 -c 0      # ECC errors
npu-smi info -t usages -i 0 -c 0   # Utilization
npu-smi info -t pcie-info -i 0 -c 0 # PCIe info
npu-smi info -t product -i 0 -c 0  # Product info
```

## Commands

### Query Processes

View running processes on NPU.

```bash
npu-smi info proc -i <id> -c <chip_id>
```

**Note:** Not supported on all platforms (e.g., Ascend 910B).

**Output:**
| Field | Description |
|-------|-------------|
| PID | Process ID |
| Process Name | Application name |
| Memory Usage | Memory used |
| AI Core Usage | AI Core utilization |

### Query ECC Errors

```bash
npu-smi info -t ecc -i <id> -c <chip_id>
```

**Output:**
- ECC Error Count
- ECC Mode (Enabled/Disabled)

### Query Utilization

```bash
npu-smi info -t usages -i <id> -c <chip_id>
```

**Output:**
- AI Core Usage (%)
- Memory Usage (%)
- Bandwidth Usage (%)

### Query PCIe Info

```bash
npu-smi info -t pcie-info -i <id> -c <chip_id>
```

**Output:**
- PCIe Speed (GT/s)
- PCIe Width (x16, x8, etc.)

### Query P2P Status

```bash
npu-smi info -t p2p -i <id> -c <chip_id>
```

**Output:**
- P2P Status
- P2P Mode

### Query Product Info

```bash
npu-smi info -t product -i <id> -c <chip_id>
```

**Output:**
- Product Name
- Product Serial Number

## Examples

### Check for Errors

```bash
#!/bin/bash

NPU=0
CHIP=0

echo "=== Error Check ==="
echo "ECC Errors:"
npu-smi info -t ecc -i $NPU -c $CHIP

echo ""
echo "Health Status:"
npu-smi info -t health -i $NPU
```

### Full System Report

```bash
#!/bin/bash

NPU=0
CHIP=0

echo "=== Advanced System Report ==="
echo ""
echo "Processes:"
npu-smi info proc -i $NPU -c $CHIP 2>/dev/null || echo "Process info not available"

echo ""
echo "ECC Status:"
npu-smi info -t ecc -i $NPU -c $CHIP

echo ""
echo "Utilization:"
npu-smi info -t usages -i $NPU -c $CHIP

echo ""
echo "PCIe Info:"
npu-smi info -t pcie-info -i $NPU -c $CHIP

echo ""
echo "Product Info:"
npu-smi info -t product -i $NPU -c $CHIP
```

## Related Skills

- [../basic/](basic/SKILL.md) - Device listing and basic info
- [../metrics/](metrics/SKILL.md) - Temperature, power, memory
- [../../npu-smi-clear/](npu-smi-config/clear/SKILL.md) - Clear ECC errors
