# Evaluation report

## Overall metrics across conditions

| Condition | Contracts | Precision (micro) | Recall (micro) | F1 (micro) | F1 (macro) |
|---|---|---|---|---|---|
| A: Baseline (single prompt) | 10 | 0.723 | 0.531 | 0.613 | 0.654 |
| C: Full system (orchestrator + validation) | 10 | 0.632 | 0.621 | 0.626 | 0.631 |

## Per-category F1

| Category | F1 (A) | F1 (C) |
|---|---|---|
| Governing Law | 0.889 | 0.783 |
| IP Ownership Assignment | 0.400 | 0.375 |
| Liability Cap | 0.727 | 0.732 |
| Termination for Convenience | 0.600 | 0.632 |

## Detail — Condition A

| Category | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Governing Law | 8 | 0 | 2 | 1.000 | 0.800 | 0.889 |
| IP Ownership Assignment | 8 | 8 | 16 | 0.500 | 0.333 | 0.400 |
| Liability Cap | 12 | 1 | 8 | 0.923 | 0.600 | 0.727 |
| Termination for Convenience | 6 | 4 | 4 | 0.600 | 0.600 | 0.600 |

## Detail — Condition C

| Category | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Governing Law | 9 | 4 | 1 | 0.692 | 0.900 | 0.783 |
| IP Ownership Assignment | 6 | 11 | 9 | 0.353 | 0.400 | 0.375 |
| Liability Cap | 15 | 2 | 9 | 0.882 | 0.625 | 0.732 |
| Termination for Convenience | 6 | 4 | 3 | 0.600 | 0.667 | 0.632 |
