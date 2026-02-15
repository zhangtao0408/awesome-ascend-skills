---
name: npu-smi-info-metrics
description: npu-smi metrics queries for Huawei Ascend NPU. Use when monitoring temperature, power, memory, frequency, and sensor data. Covers real-time performance metrics.
---

# npu-smi Metrics Queries

Query device metrics using `npu-smi info`.

## Quick Reference

```bash
npu-smi info -t temp -i 0 -c 0    # Temperature
npu-smi info -t power -i 0 -c 0   # Power
npu-smi info -t memory -i 0 -c 0  # Memory
npu-smi info -t freq -i 0 -c 0    # Frequency
npu-smi info -t sensors -i 0 -c 0 # Sensors
```

## Commands

### Query Temperature

```bash
npu-smi info -t temp -i <id> -c <chip_id>
```

**Output:**
- NPU Temperature (°C)
- AI Core Temperature (°C)

**Example:**
```bash
$ npu-smi info -t temp -i 0 -c 0
Device 0, Chip 0:
    NPU Temperature     : 45 C
    AI Core Temperature : 48 C
```

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

### Query Frequency

```bash
npu-smi info -t freq -i <id> -c <chip_id>
```

**Output:**
- AI Core Frequency (MHz)
- Memory Frequency (MHz)

### Query Sensors

```bash
npu-smi info -t sensors -i <id> -c <chip_id>
```

**Output:**
- Temperature sensors
- Voltage readings
- Current readings

## Examples

### Temperature Monitor

```bash
#!/bin/bash

while true; do
    clear
    echo "=== Temperature Monitor $(date) ==="
    npu-smi info -l | grep -E '^\|\s+[0-9]+' | while read line; do
        npu=$(echo $line | awk '{print $2}')
        temp=$(npu-smi info -t temp -i $npu -c 0 | grep "NPU Temperature" | awk '{print $4}')
        echo "NPU $npu: ${temp}°C"
    done
    sleep 5
done
```

### Resource Usage Report

```bash
#!/bin/bash

NPU=0
CHIP=0

echo "=== Resource Usage Report ==="
echo ""
echo "Temperature:"
npu-smi info -t temp -i $NPU -c $CHIP

echo ""
echo "Power:"
npu-smi info -t power -i $NPU -c $CHIP

echo ""
echo "Memory:"
npu-smi info -t memory -i $NPU -c $CHIP

echo ""
echo "Frequency:"
npu-smi info -t freq -i $NPU -c $CHIP
```

## Related Skills

- [../basic/](basic/SKILL.md) - Device listing and basic info
- [../advanced/](advanced/SKILL.md) - Processes, ECC, utilization
- [../../npu-smi-config/thresholds/](npu-smi-config/thresholds/SKILL.md) - Set thresholds
