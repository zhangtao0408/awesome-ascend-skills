---
name: npu-smi-info-basic
description: npu-smi basic info queries for Huawei Ascend NPU. Use when listing devices, checking device health, or viewing board and chip information. Covers device discovery and basic status checks.
---

# npu-smi Basic Info Queries

Query basic device information using `npu-smi info`.

## Quick Reference

```bash
npu-smi info -l              # List all devices
npu-smi info -t health -i 0  # Check device health
npu-smi info -t board -i 0   # View board info
npu-smi info -t npu -i 0 -c 0 # View chip info
npu-smi info -m              # List all chips
```

## Commands

### List Devices

List all NPU devices in the system.

```bash
npu-smi info -l
```

**Output:**
| Field | Description |
|-------|-------------|
| NPU ID | Device identifier |
| Name | Device name |

**Example:**
```bash
$ npu-smi info -l
+-----------+-----------+
| NPU ID    | Name      |
+-----------+-----------+
| 0         | 910B      |
| 1         | 910B      |
+-----------+-----------+
```

### Query Device Health

Check health status of a specific NPU.

```bash
npu-smi info -t health -i <id>
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | integer | Yes | Device ID from `info -l` |

**Output:**
| Field | Values |
|-------|--------|
| Healthy | OK, Warning, Error |

### Query Board Information

View detailed board info including firmware version.

```bash
npu-smi info -t board -i <id>
```

**Output:**
| Field | Description |
|-------|-------------|
| NPU ID | Device identifier |
| Name | Board name |
| Health | Health status |
| Power Usage | Current power draw |
| Temperature | Board temperature |
| Firmware Version | Current firmware |
| Software Version | Driver version |

### Query NPU/Chip Details

View chip-level details.

```bash
npu-smi info -t npu -i <id> -c <chip_id>
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | integer | Yes | Device ID |
| chip_id | integer | Yes | Chip ID from `info -m` |

**Output:**
| Field | Description |
|-------|-------------|
| Chip ID | Chip identifier |
| Name | Chip name |
| Health | Health status |
| Power Usage | Power consumption |
| Temperature | Chip temperature |
| Memory Usage | Memory utilization |
| AI Core Usage | AI Core utilization |

### List All Chips

Query summary of all chips.

```bash
npu-smi info -m
```

**Output:**
| Field | Description |
|-------|-------------|
| NPU ID | Parent device |
| Chip ID | Chip identifier |
| Name | Chip name |
| Health | Health status |

## Examples

### Quick Health Check

```bash
#!/bin/bash

# Check health of all devices
npu-smi info -l | grep -E '^\|\s+[0-9]+' | while read line; do
    npu=$(echo $line | awk '{print $2}')
    health=$(npu-smi info -t health -i $npu | grep Healthy | awk '{print $2}')
    echo "NPU $npu: $health"
done
```

### Get Device Summary

```bash
#!/bin/bash

NPU=0

echo "=== Device $NPU Summary ==="
npu-smi info -t health -i $NPU
npu-smi info -t board -i $NPU
```

## Related Skills

- [../npu-smi-info/metrics/](metrics/SKILL.md) - Temperature, power, memory queries
- [../npu-smi-info/advanced/](advanced/SKILL.md) - Processes, ECC, utilization
- [../../npu-smi-config/](npu-smi-config/SKILL.md) - Configuration commands
