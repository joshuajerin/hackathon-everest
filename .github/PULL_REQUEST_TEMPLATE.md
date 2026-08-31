## Summary

## Validation

- [ ] `make verify`
- [ ] `make test-isaac-unit` when extension tensor/runtime code changed
- [ ] `git diff --check`

## Evidence and claim boundary

Describe what this change proves, what remains simulator-only, and any asset/policy/checkpoint hashes needed to reproduce it.

## Contract checklist

- [ ] Per-foot packet remains exactly 19 values
- [ ] Validity, age, context, labels, and privileged truth remain separate
- [ ] Safety and progress are reported together
- [ ] No credentials, proprietary checkpoints, generated datasets, or unlicensed assets are included
