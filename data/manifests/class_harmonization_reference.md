# Class Harmonization Reference

This file is the working label reference for the two prepared datasets in Objective 1.

Current decision:

Keep `PKU COCO` and `DeepPCB COCO` separate for baseline use. Do not merge their labels yet.

## Baseline Label Strategy

### PKU COCO baseline

Use the PKU dataset with its own six training classes:

1. `missing_hole`
2. `mouse_bite`
3. `open_circuit`
4. `short`
5. `spur`
6. `spurious_copper`

Important:

- Ignore category id `0` = `pcb-defects` during training and evaluation.
- Keep PKU class names unchanged for PKU-only experiments.

### DeepPCB COCO baseline

Use the DeepPCB dataset with its own six training classes:

1. `open`
2. `short`
3. `mousebite`
4. `spur`
5. `copper`
6. `pin-hole`

Important:

- Preserve the original DeepPCB ids `1..6`.
- Keep DeepPCB class names unchanged for DeepPCB-only experiments.

## Harmonization Table

| PKU class | DeepPCB class | Match type | Current handling |
| --- | --- | --- | --- |
| `short` | `short` | Exact match | Same defect meaning; safe to compare conceptually |
| `spur` | `spur` | Exact match | Same defect meaning; safe to compare conceptually |
| `open_circuit` | `open` | Exact semantic match with naming difference | Treat as equivalent in reports, but keep native names in each dataset |
| `mouse_bite` | `mousebite` | Exact semantic match with naming difference | Treat as equivalent in reports, but keep native names in each dataset |
| `spurious_copper` | `copper` | Near match | Likely related, but keep separate until canonical naming is approved |
| `missing_hole` | `pin-hole` | Unresolved pair | Must stay separate for now |

## Practical Interpretation

### Exact matches

- `short` <-> `short`
- `spur` <-> `spur`

### Exact semantic matches with naming differences

- `open_circuit` <-> `open`
- `mouse_bite` <-> `mousebite`

### Near matches

- `spurious_copper` <-> `copper`

### Unresolved pairs that must stay separate for now

- `missing_hole` <-> `pin-hole`

## Rule To Use Later

Until a future label-unification step is explicitly approved:

- train PKU with PKU labels only
- train DeepPCB with DeepPCB labels only
- compare results at the dataset level, not through a merged label map
- do not create a shared six-class training target yet
