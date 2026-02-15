---
name: npu-smi-cert-monitor
description: npu-smi certificate monitoring for Huawei Ascend NPU. Use when checking certificate expiration and monitoring certificate health.
---

# npu-smi Certificate Monitoring

Monitor certificate expiration and health.

## Quick Reference

```bash
npu-smi info -t tls-cert -i 0 -c 0 | grep "Valid Until"
npu-smi info -t tls-cert-period -i 0 -c 0
```

## Monitoring Certificate Expiration

```bash
#!/bin/bash

NPU=0
CHIP=0

# Get expiration date
EXPIRY=$(npu-smi info -t tls-cert -i $NPU -c $CHIP | grep "Valid Until" | cut -d: -f2-)
echo "Certificate expires: $EXPIRY"

# Get threshold
THRESHOLD=$(npu-smi info -t tls-cert-period -i $NPU -c $CHIP | grep -o '[0-9]*')
echo "Warning threshold: $THRESHOLD days"
```

## Expiration Alert Script

```bash
#!/bin/bash

THRESHOLD=30

echo "=== Certificate Expiration Check ==="

npus=$(npu-smi info -l | grep -E '^\|\s+[0-9]+' | awk '{print $2}')

for npu in $npus; do
    chips=$(npu-smi info -m | grep "^| $npu " | awk '{print $4}')
    for chip in $chips; do
        cert_info=$(npu-smi info -t tls-cert -i $npu -c $chip 2>/dev/null)
        if [ $? -eq 0 ]; then
            expiry=$(echo "$cert_info" | grep "Valid Until" | cut -d: -f2- | xargs)
            if [ -n "$expiry" ]; then
                expiry_ts=$(date -d "$expiry" +%s 2>/dev/null || echo "0")
                now_ts=$(date +%s)
                days_left=$(( (expiry_ts - now_ts) / 86400 ))
                
                if [ $days_left -lt $THRESHOLD ]; then
                    echo "WARNING: NPU $npu, Chip $chip expires in $days_left days"
                else
                    echo "OK: NPU $npu, Chip $chip - $days_left days remaining"
                fi
            fi
        fi
    done
done
```

## Related Skills

- [../query/](query/SKILL.md) - Query certificate info
- [../manage/](manage/SKILL.md) - Import new certificates
