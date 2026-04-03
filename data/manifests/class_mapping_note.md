# Dataset Class Mapping Note

This note compares the current PCB defect class labels in the two prepared datasets and records how we should treat them before tiling or training.

## Dataset Labels

### PKU COCO

Usable classes:

| Id | Name |
| --- | --- |
| 1 | `missing_hole` |
| 2 | `mouse_bite` |
| 3 | `open_circuit` |
| 4 | `short` |
| 5 | `spur` |
| 6 | `spurious_copper` |

Note:

`pcb-defects` with id `0` exists only as a parent category in the Roboflow export and should not be used as a training class.

### DeepPCB COCO

Usable classes:

| Id | Name |
| --- | --- |
| 1 | `open` |
| 2 | `short` |
| 3 | `mousebite` |
| 4 | `spur` |
| 5 | `copper` |
| 6 | `pin-hole` |

## Cross-Dataset Relationship

| PKU class | DeepPCB class | Relationship | Practical note |
| --- | --- | --- | --- |
| `open_circuit` | `open` (id 1) | Very likely same concept | Naming difference only |
| `short` | `short` (id 2) | Same | Safe match |
| `mouse_bite` | `mousebite` (id 3) | Same | Naming difference only |
| `spur` | `spur` (id 4) | Same | Safe match |
| `spurious_copper` | `copper` (id 5) | Likely same or near-equivalent | Needs explicit naming decision before merging |
| `missing_hole` | `pin-hole` (id 6) | Related but not clearly identical | Do not merge automatically without confirmation |

## Decision For Now

Keep the datasets separate for now.

Reason:

- Four class pairs are straightforward naming matches: `open/open_circuit`, `short`, `mousebite/mouse_bite`, and `spur`.
- Two class pairs still need an explicit harmonization decision: `copper` vs `spurious_copper`, and `pin-hole` vs `missing_hole`.
- Because of those unresolved pairs, combining the datasets now would risk introducing label noise.

## Recommended Later Unified Label Map

If the project later combines the datasets, create an approved unified label map first.

Minimum safe candidate map:

1. `open_circuit`
2. `short`
3. `mouse_bite`
4. `spur`
5. `spurious_copper` or `copper` (pick one canonical name first)
6. hole-defect class only after confirming whether `pin-hole` and `missing_hole` should be treated as the same label
