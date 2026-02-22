---
name: npu-smi-info-basic
description: npu-smi basic info queries for Huawei Ascend NPU. Use when listing devices, checking device health, viewing board and chip information, or monitoring temperature, power, and memory metrics. Covers device discovery, basic status checks, and real-time performance metrics.
---

# npu-smi Basic Info Queries

Query basic device information and real-time metrics using `npu-smi info`.

## Quick Reference

```bash
npu-smi info -l              # List all devices
npu-smi info -t health -i 0  # Check device health
npu-smi info -t board -i 0   # View board info
npu-smi info -t npu -i 0 -c 0 # View chip info
npu-smi info -m              # List all chips
npu-smi info -t temp -i 0 -c 0   # Temperature
npu-smi info -t power -i 0 -c 0  # Power
npu-smi info -t memory -i 0 -c 0 # Memory
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

**Note:** Output format may vary by npu-smi version and hardware platform.

**Example (simplified):**
```bash
$ npu-smi info -l
Total        : 8 NPU in system
NPU          : 0
Name         : 910B3
...
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

### Query Temperature

```bash
npu-smi info -t temp -i <id> -c <chip_id>
```

**Output:**
- NPU Temperature (°C)
- AI Core Temperature (°C)

**Note:** Output format may vary by npu-smi version.

### Query Power

```bash
npu-smi info -t power -i <id> -c <chip_id>
```

**Output:**
- Power Usage (W)
- Power Limit (W)

### Query Memory

```bash
npu-smi info -t memory -i <id> -c <chip_id>
```

**Output:**
- Memory Usage (MB)
- Memory Total (MB)
- Memory Usage Rate (%)

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
CHIP=0

echo "=== Device $NPU Summary ==="
npu-smi info -t health -i $NPU
npu-smi info -t board -i $NPU
echo ""
echo "=== Metrics ==="
npu-smi info -t temp -i $NPU -c $CHIP
npu-smi info -t power -i $NPU -c $CHIP
npu-smi info -t memory -i $NPU -c $CHIP
```

### Resource Monitoring Script

```bash
#!/bin/bash

echo "=== NPU Resource Monitor $(date) ==="

for npu in $(npu-smi info -l 2>/dev/null | grep -oP 'NPU\s*:\s*\K[0-9]+'); do
    echo ""
    echo "--- NPU $npu ---"
    temp=$(npu-smi info -t temp -i $npu -c 0 2>/dev/null | grep -oP 'NPU Temperature\s*:\s*\K[0-9]+' || echo "N/A")
    power=$(npu-smi info -t power -i $npu -c 0 2>/dev/null | grep -oP 'Power Usage\s*:\s*\K[0-9.]+' || echo "N/A")
    mem=$(npu-smi info -t memory -i $npu -c 0 2>/dev/null | grep -oP 'Memory Usage Rate\s*:\s*\K[0-9]+' || echo "N/A")
    echo "  Temperature: ${temp}°C"
    echo "  Power: ${power}W"
    echo "  Memory Usage: ${mem}%"
done
```

## Related Skills

- [../npu-smi-info/advanced/](advanced/SKILL.md) - Processes, ECC, utilization
- [../../npu-smi-config/thresholds/](npu-smi-config/thresholds/SKILL.md) - Set temperature/power thresholds
