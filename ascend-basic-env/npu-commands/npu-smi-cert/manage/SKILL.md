---
name: npu-smi-cert-manage
description: npu-smi certificate management commands for Huawei Ascend NPU. Use when importing TLS certificates or setting expiration thresholds.
---

# npu-smi Certificate Management

Manage certificates using `npu-smi set` and `npu-smi clear`.

## Quick Reference

```bash
npu-smi set -t tls-cert -i 0 -c 0 -f "tls.pem ca.pem sub.pem"
npu-smi set -t tls-cert-period -i 0 -c 0 -s 90
npu-smi clear -t tls-cert-period -i 0 -c 0
```

## Commands

### Import TLS Certificate

```bash
npu-smi set -t tls-cert -i <id> -c <chip_id> -f "<tls> <ca> <subca>"
```

**Files:** PEM format certificates

### Set Expiration Threshold

```bash
npu-smi set -t tls-cert-period -i <id> -c <chip_id> -s <days>
```

**Range:** [7-180] days, default 90

### Restore Default Threshold

```bash
npu-smi clear -t tls-cert-period -i <id> -c <chip_id>
```

## Examples

### Import Certificates

```bash
#!/bin/bash

NPU=0
CHIP=0

echo "Importing certificates..."
npu-smi set -t tls-cert -i $NPU -c $CHIP \
  -f "device.pem ca-root.pem ca-sub.pem"

echo "Setting 60-day threshold..."
npu-smi set -t tls-cert-period -i $NPU -c $CHIP -s 60

echo "Done!"
```

## Related Skills

- [../query/](query/SKILL.md) - View certificates
- [../monitor/](monitor/SKILL.md) - Monitor expiration
