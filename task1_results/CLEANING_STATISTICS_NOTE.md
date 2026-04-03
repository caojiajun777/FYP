# Cleaning Statistics — Scope Note

`cleaning_statistics.json` records a **second-pass cleaning run** applied to an intermediate dataset
state (before/after: 3247 → 3230 total).

## This file does NOT represent the main cleaning claim

The primary dissertation claim is:

| Step | Count |
|------|-------|
| Raw corpus | 4320 |
| After full leakage + duplicate removal | **3546** |
| Protocol A fixed split | **2913 / 316 / 317** |

The evidence file for that claim is: `data_cleaning_impact_report.md`

## What this file actually records

A late incremental pass that removed 17 internal train-set duplicates from a dataset that was
already partially cleaned (train=2930, test=317). This resulted in the final Protocol A
train split of 2913:

| Split | Before this pass | After this pass |
|-------|-----------------|-----------------|
| Train | 2930 | **2913** (−17 duplicates) |
| Val   | 0    | 0 |
| Test  | 317  | 317 |
| Total | 3247 | 3230 |

Note: at this stage the val set had already been merged into the train pool (Protocol A uses
a fixed train/test only, with a separate val drawn from the cleaned pool).

## How deduplication was detected

Perceptual hashing (aHash + pHash) with Hamming distance = 0 (exact visual duplicates).
For full methodology see `task1_source_classification/analysis/data_quality_verification.md`.
