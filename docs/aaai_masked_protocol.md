# AAAI Masked Robustness Protocol

This document turns the paper plan into a repeatable experiment protocol.
The goal is to stop ad-hoc checkpoint selection and keep the masked-robustness
story decision-complete for the paper.

## 1. Screening experiments

All screening experiments inherit from `exp26`, start from
`work_dirs/sparsedrive_small_stage2_exp27/iter_42195.pth`, keep the masked
curriculum fixed, and run for `14064` iterations with evaluation every `7032`
iterations.

Configs:

- `projects/configs/aaai/sparsedrive_small_stage2_aaai_e1_planning_only.py`
- `projects/configs/aaai/sparsedrive_small_stage2_aaai_e2_temporal_only.py`
- `projects/configs/aaai/sparsedrive_small_stage2_aaai_e3_pvrecon_only.py`
- `projects/configs/aaai/sparsedrive_small_stage2_aaai_e4_temporal_pvrecon.py`

Freeze policy:

- `E1`: freeze `pv_recon` and `temporal_completion`; adapt planning-related modules only
- `E2`: unfreeze `temporal_completion` only
- `E3`: unfreeze `pv_recon` only
- `E4`: unfreeze both `temporal_completion` and `pv_recon`

Run training:

```bash
bash ./tools/dist_train.sh \
    projects/configs/aaai/sparsedrive_small_stage2_aaai_e1_planning_only.py \
    1 \
    --deterministic
```

Replace the config path with `E2` / `E3` / `E4` as needed.

## 2. Dual evaluation

Each checkpoint must be evaluated twice: once with standard eval and once with
masked eval. Do not use `latest.pth` as the paper result.

```bash
bash ./scripts/run_aaai_dual_eval.sh \
    projects/configs/aaai/sparsedrive_small_stage2_aaai_e1_planning_only.py \
    work_dirs/sparsedrive_small_stage2_aaai_e1_planning_only/iter_7032.pth \
    1
```

The script writes to:

```text
<experiment_dir>/evals/<checkpoint_name>/standard/
<experiment_dir>/evals/<checkpoint_name>/masked/
```

Each directory contains `e2e_metrics.json`, `metrics_summary.json`, and
`results.pkl`.

## 3. Metric collection and checkpoint selection

Use the collector after each screening stage. It picks:

- `best_masked_l2`: lowest masked `L2`
- `best_masked_safe`: lowest masked `obj_box_col`
- `best_standard`: highest standard `NDS`

Example:

```bash
python ./tools/collect_aaai_metrics.py \
    work_dirs/sparsedrive_small_stage2_aaai_e1_planning_only \
    work_dirs/sparsedrive_small_stage2_aaai_e2_temporal_only \
    --baseline-standard /path/to/baseline_standard/e2e_metrics.json \
    --baseline-masked /path/to/baseline_masked/e2e_metrics.json \
    --json-out /tmp/aaai_masked_summary.json
```

Paper thresholds:

- masked `L2` should improve by at least `0.005` over the masked baseline
- masked `obj_box_col` should stay at or below `0.13%`
- standard `NDS` should not drop by more than `0.015`
- standard `L2` should not worsen by more than `0.02`

## 4. Continue the winner

Only extend one winning route to `28130` iterations. Do not extend all four.

```bash
bash ./scripts/continue_aaai_winner.sh \
    projects/configs/aaai/sparsedrive_small_stage2_aaai_e2_temporal_only.py \
    work_dirs/sparsedrive_small_stage2_aaai_e2_temporal_only/iter_14064.pth \
    work_dirs/sparsedrive_small_stage2_aaai_e2_temporal_only_long \
    1 \
    28130
```

This preserves the config, swaps in the chosen checkpoint, and keeps the
evaluation interval fixed at `7032`.

## 5. Paper-facing defaults

- Primary claim: masked robustness under camera missing
- Primary metric: masked `L2`
- Safety metric: masked `obj_box_col`
- Standard eval is a guardrail, not the optimization target
- `exp28` should stay as a negative-control result, not the main method
