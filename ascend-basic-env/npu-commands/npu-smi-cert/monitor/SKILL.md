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

npus=$(npu-smi info -l 2>/dev/null | grep -oP 'NPU\s*:\s*\K[0-9]+' || echo "")

if [ -z "$npus" ]; then
    echo "ERROR: No NPU devices found or npu-smi not available"
    exit 1
fi

for npu in $npus; do
    chips=$(npu-smi info -m 2>/dev/null | grep "NPU $npu" | grep -oP 'Chip\s*:\s*\K[0-9]+' || echo "")
    if [ -z "$chips" ]; then
        echo "INFO: NPU $npu - No chips found"
        continue
    fi
    
    for chip in $chips; do
        cert_info=$(npu-smi info -t tls-cert -i $npu -c $chip 2>/dev/null)
        
        # Check if certificate query failed or no certificate exists
        if [ $? -ne 0 ]; then
            echo "INFO: NPU $npu, Chip $chip - Certificate query failed or not supported"
            continue
        fi
        
        # Check for "No certificate" or empty response
        if echo "$cert_info" | grep -qi "no certificate\|not found\|none"; then
            echo "INFO: NPU $npu, Chip $chip - No certificate installed"
            continue
        fi
        
        expiry=$(echo "$cert_info" | grep -i "Valid Until" | cut -d: -f2- | xargs)
        
        if [ -z "$expiry" ]; then
            echo "INFO: NPU $npu, Chip $chip - Could not parse certificate expiration"
            continue
        fi
        
        # Parse expiration date (handle different date formats)
        expiry_ts=$(date -d "$expiry" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M:%S" "$expiry" +%s 2>/dev/null || echo "0")
        
        if [ "$expiry_ts" = "0" ]; then
            echo "WARN: NPU $npu, Chip $chip - Could not parse expiration date: $expiry"
            continue
        fi
        
        now_ts=$(date +%s)
        days_left=$(( (expiry_ts - now_ts) / 86400 ))
        
        if [ $days_left -lt 0 ]; then
            echo "CRITICAL: NPU $npu, Chip $chip - Certificate EXPIRED $((-days_left)) days ago"
        elif [ $days_left -lt $THRESHOLD ]; then
            echo "WARNING: NPU $npu, Chip $chip expires in $days_left days"
        else
            echo "OK: NPU $npu, Chip $chip - $days_left days remaining"
        fi
    done
done
```

## Related Skills

- [../query/](query/SKILL.md) - Query certificate info
- [../manage/](manage/SKILL.md) - Import new certificates
