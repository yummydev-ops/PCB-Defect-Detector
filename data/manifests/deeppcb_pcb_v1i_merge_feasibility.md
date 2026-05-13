# DeepPCB vs Cleaned PCB v1i Merge Feasibility

This note checks whether the current `DeepPCB` dataset and the cleaned `PCB-Defect-Detection.v1i.coco` dataset can be merged later without damaging label consistency.

## Harmonization Table

| Cleaned PCB v1i class | DeepPCB class | Relationship | Merge status |
| --- | --- | --- | --- |
| `short` | `short` | Exact match | Safe |
| `spur` | `spur` | Exact match | Safe |
| `open_circuit` | `open` | Same defect concept, different name | Likely safe after renaming |
| `mouse_bite` | `mousebite` | Same defect concept, different name | Likely safe after renaming |
| `spurious_copper` | `copper` | Near match, but not clearly identical from name alone | Not clean yet |
| `missing_hole` | `pin-hole` | Related but not clearly identical | Not clean yet |

## Practical Reading

- Four pairs are straightforward or near-straightforward harmonization cases:
  - `short`
  - `spur`
  - `open_circuit` <-> `open`
  - `mouse_bite` <-> `mousebite`
- Two pairs still create label-risk:
  - `spurious_copper` <-> `copper`
  - `missing_hole` <-> `pin-hole`

## Merge Feasibility

- A combined dataset is **feasible later**, but it is **not cleanly feasible right now** without an explicit label-harmonization decision.
- If we merge now, we risk injecting label noise into the final training set because two class pairs do not map cleanly enough yet.

## Minimum Steps Before Any Future Merge

1. Approve one canonical class name set for the merged dataset.
2. Manually resolve `spurious_copper` vs `copper`.
3. Manually resolve `missing_hole` vs `pin-hole`.
4. Remap both datasets into the approved shared label set while preserving a record of the original source labels.
5. Recheck class counts and visually spot-check a few remapped samples before training.

## Clear Conclusion

- **No**: a combined DeepPCB + cleaned PCB v1i training set should **not** be built next.
- The label space is close, but not clean enough yet for a safe merge.
