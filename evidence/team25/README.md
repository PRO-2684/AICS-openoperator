# team25 Evidence Package

This directory contains the organized evidence package for the audit of `cotangents/openoperator-start-kit-team25`.

## Reports

- `report_en.md`: English report.
- `report_zh.md`: Chinese report.
- `report_zh_draft_old.md`: old translated draft kept for traceability.

## Machine-Readable Evidence

- `exact_blob_matches.tsv`: exact git blob matches across the two repositories.
- `head_similarity_matches.tsv`: normalized-token similarity matches from current team25 HEAD to earlier MosRat history.

## Raw Evidence

- `raw/team25_suspicious_commit_stats.txt`
- `raw/masked_softmax_blob_proof.txt`
- `raw/team25_mlu_log.tsv`
- `raw/ours_mlu_log.tsv`

## Diff Evidence

- `diffs/Grid_sample_ours_5fe8d26e7_vs_team25_HEAD.diff`
- `diffs/Matmul_with_irregular_shapes_ours_af7f6074b_vs_team25_HEAD.diff`
- `diffs/Matmul_with_large_K_dimension_ours_af7f6074b_vs_team25_HEAD.diff`
- `diffs/Masked_softmax_ours_c1573f279_vs_team25_HEAD.diff` is intentionally empty because the compared files have no diff.
