# ARDA: Pre-Aggregation + Feature Selection

This module documents how we implement the ARDA framework (A Robust Data Augmentation Framework for Tabular Data). Because our relational tables already carry explicit primary/foreign-key links, we can skip the coreset-based join discovery described in the original paper and focus on transforming and selecting features.

## Pre-Aggregation

> “To address the issue with one-to-many and many-to-many join we pre-aggregate foreign tables on join keys, thereby effectively reducing to the one-to-one and many-to-one cases.”

Our implementation aggregates one-hop foreign tables only, yielding `many-to-one`, `one-to-many`, and the special `one-to-one` case (joined directly without aggregation).

Supported aggregation functions:
- Numerical columns: `mean`, `sum`, `max`, `min`, `count`
- Categorical columns: `mode`, `count`
- Text columns: ignored (no aggregation)
- Datetime columns: `min`, `max`

> Temporal features are ignored, consistent with the AIDA paper.

## Feature Selection

The ARDA paper proposes Random Feature Injection (RFI), a noise-driven feature selection routine geared toward data-lake scenarios where spurious joins are common.

Traditional feature selection tends to be:
- Sensitive to noisy, high-dimensional inputs
- Computationally expensive
- Prone to having true signals masked by irrelevant features

RFI mitigates these issues by comparing real features against injected random features:
1. Inject random features using Algorithm 2 from the paper.
2. Train an ensemble of base learners (RandomForest + Sparse Regression) and rank real and random features together by importance.
3. Repeat steps 1–2 for *K* trials to stabilize the rankings.
4. For each real feature, compute how often it appears before every random feature and normalize the frequency into a confidence score.
5. Select features whose confidence exceeds a chosen threshold.

RFI requires a downstream predictive task to evaluate feature importance.

### Current implementation gap

Today we enumerate one-hop neighbor tables with DFS, generate candidate features, and rank them using a simple correlation heuristic. Integrating full RFI support remains future work.

## References
- [ARDA: A Robust Data Augmentation Framework for Tabular Data](https://arxiv.org/abs/2306.07330)
