# Normalized Severity Scoring Method

This note defines the first practical severity score used with the frozen final PCB defect detector.

## What The Score Means

- The severity score is a **heuristic prioritization score** for detected defects.
- It is **not** expert ground-truth damage grading.
- Its main purpose is to make later reports and UI outputs easier to sort, rank, and review.

## Score Range

- Severity score range: `0.0` to `1.0`
- Lower scores mean lower relative concern in the current heuristic.
- Higher scores mean higher relative concern in the current heuristic.

## Scoring Rule

For each detection:

```text
severity_score =
  0.50 * class_prior
  + 0.30 * area_score
  + 0.20 * confidence
```

Then clamp the final result to the range `0.0` to `1.0`.

### Class Priors

- `missing_hole`: `0.70`
- `mouse_bite`: `0.55`
- `open_circuit`: `1.00`
- `short`: `1.00`
- `spur`: `0.60`
- `spurious_copper`: `0.75`

### Area Score

- Compute relative defect size:

```text
relative_area = bbox_area / image_area
```

- Normalize it with a simple reference area of `1%` of the image:

```text
area_score = min(1.0, relative_area / 0.01)
```

This means larger detected defects contribute more strongly to severity, but the area contribution is capped at `1.0`.

### Confidence

- Use the detector confidence directly after clipping to `0.0` to `1.0`.

## Severity Bands

- `low`: score `< 0.34`
- `medium`: score `0.34` to `< 0.67`
- `high`: score `>= 0.67`

## How It Appears In Outputs

The structured inference output now records, for each detection:

- defect class
- confidence
- bounding box
- relative area
- severity score
- severity band

## How It Will Be Used In Reports

- Reports can sort detections from highest to lowest severity.
- Reports can highlight high-severity detections first.
- The raw score stays visible so the scoring remains transparent and reproducible.
