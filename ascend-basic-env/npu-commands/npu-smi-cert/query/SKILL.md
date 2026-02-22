---
name: npu-smi-cert-query
description: npu-smi certificate query commands for Huawei Ascend NPU. Use when viewing certificate info, checking expiration, or generating CSR.
---

# npu-smi Certificate Queries

Query certificate information using `npu-smi info`.

## Quick Reference

```bash
npu-smi info -t tls-csr-get -i 0 -c 0      # Generate CSR
npu-smi info -t tls-cert -i 0 -c 0         # View certificate
npu-smi info -t tls-cert-period -i 0 -c 0  # Check threshold
npu-smi info -t rootkey -i 0 -c 0          # Rootkey status
```

## Commands

### Get CSR

```bash
npu-smi info -t tls-csr-get -i <id> -c <chip_id>
```

**Output:** CSR in PEM format

### View Certificate

```bash
npu-smi info -t tls-cert -i <id> -c <chip_id>
```

**Output:**
- Subject
- Issuer
- Valid From/Until
- Serial Number

### Query Expiration Threshold

```bash
npu-smi info -t tls-cert-period -i <id> -c <chip_id>
```

**Output:** Days before warning

### Query Rootkey Status

```bash
npu-smi info -t rootkey -i <id> -c <chip_id>
```

## Examples

### Check Certificate Status

```bash
#!/bin/bash

NPU=0
CHIP=0

echo "Certificate Info:"
npu-smi info -t tls-cert -i $NPU -c $CHIP

echo ""
echo "Expires in threshold:"
npu-smi info -t tls-cert-period -i $NPU -c $CHIP
```

## Related Skills

- [../manage/](manage/SKILL.md) - Import certificates
- [../monitor/](monitor/SKILL.md) - Monitor expiration
