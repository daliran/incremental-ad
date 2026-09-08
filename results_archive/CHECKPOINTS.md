# Checkpoints — what to preserve, and how to verify a copy

`results_archive/` holds the *evidence* (CSVs, `config.json`, every `result.json`) and deliberately excludes `*.pt`: they are gigabytes and reproducible from the configs. But re-merging, geometry, Fisher and any BECAME resampling need the weights, and `$WORK` is scratch. This is the record that makes an off-cluster copy verifiable.

**Scope.** 181 experiment groups that own checkpoints — every experiment named by any file in `analysis_specs/` or globbed by `regenerate_analysis.sh`, plus the `opcm2_`/`window_`/`origin_`/`basefrac_`/`selalpha_`/`n1_`/`aeft_`/`adfc2_` families. Derived, not listed, so a new spec entry cannot leave its checkpoints unbacked. **3921 files, 12.5 GB.**

## Copying

```bash
# from the cluster, one group per line so a partial copy is resumable
rsync -av --info=progress2 \
  /work/tesi_ddellacasaventurelli01/incremental-ad/runs/adfc2_psm_joint \
  /work/tesi_ddellacasaventurelli01/incremental-ad/runs/adfc2_psm_joint_oldmask \
  /work/tesi_ddellacasaventurelli01/incremental-ad/runs/adfc2_psm_merge_n2 \
  ...  # all 181 groups, listed below
  <destination>/incremental-ad-checkpoints/
```

## Verifying afterwards

```bash
python - <<'EOF'
import csv, hashlib
from pathlib import Path
root = Path('<destination>/incremental-ad-checkpoints')
bad = 0
for row in csv.DictReader(open('results_archive/checkpoints.csv')):
    p = root / row['path']
    if not p.exists():
        print('MISSING', row['path']); bad += 1; continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != row['sha256']:
        print('CORRUPT', row['path']); bad += 1
print(f'{bad} problem(s)')
EOF
```

## Groups

| experiment | checkpoints | size |
|---|---|---|
| `adfc2_psm_joint` | 6 | 0.02 GB |
| `adfc2_psm_joint_oldmask` | 6 | 0.02 GB |
| `adfc2_psm_merge_n2` | 21 | 0.06 GB |
| `adfc2_psm_merge_n2_oldmask` | 21 | 0.06 GB |
| `adfc2_psm_merge_n3` | 27 | 0.08 GB |
| `adfc2_psm_merge_n3_oldmask` | 27 | 0.08 GB |
| `adfc2_psm_merge_n5` | 39 | 0.11 GB |
| `adfc2_psm_merge_n5_oldmask` | 39 | 0.11 GB |
| `adfc2_psm_merge_tf06` | 27 | 0.08 GB |
| `adfc2_psm_merge_tf06_oldmask` | 27 | 0.08 GB |
| `adfc2_psm_merge_tf08` | 27 | 0.08 GB |
| `adfc2_psm_merge_tf08_oldmask` | 27 | 0.08 GB |
| `adfc2_psm_sequential_n2` | 18 | 0.05 GB |
| `adfc2_psm_sequential_n2_oldmask` | 18 | 0.05 GB |
| `adfc2_psm_sequential_n3` | 24 | 0.07 GB |
| `adfc2_psm_sequential_n3_oldmask` | 24 | 0.07 GB |
| `adfc2_psm_sequential_n5` | 36 | 0.11 GB |
| `adfc2_psm_sequential_n5_oldmask` | 36 | 0.11 GB |
| `adfc2_psm_window_W1` | 15 | 0.04 GB |
| `adfc2_psm_window_W1_oldmask` | 15 | 0.04 GB |
| `adfc2_psm_window_W2` | 15 | 0.04 GB |
| `adfc2_psm_window_W2_oldmask` | 15 | 0.04 GB |
| `adfc2_psm_window_W3` | 15 | 0.04 GB |
| `adfc2_psm_window_W3_oldmask` | 15 | 0.04 GB |
| `adfc2_swat_joint` | 6 | 0.02 GB |
| `adfc2_swat_joint_oldmask` | 6 | 0.02 GB |
| `adfc2_swat_merge_n2` | 21 | 0.06 GB |
| `adfc2_swat_merge_n2_oldmask` | 21 | 0.06 GB |
| `adfc2_swat_merge_n3` | 27 | 0.08 GB |
| `adfc2_swat_merge_n3_oldmask` | 27 | 0.08 GB |
| `adfc2_swat_merge_n5` | 39 | 0.12 GB |
| `adfc2_swat_merge_n5_oldmask` | 39 | 0.12 GB |
| `adfc2_swat_merge_tf06` | 27 | 0.08 GB |
| `adfc2_swat_merge_tf06_oldmask` | 27 | 0.08 GB |
| `adfc2_swat_merge_tf08` | 27 | 0.08 GB |
| `adfc2_swat_merge_tf08_oldmask` | 27 | 0.08 GB |
| `adfc2_swat_sequential_n2` | 18 | 0.05 GB |
| `adfc2_swat_sequential_n2_oldmask` | 18 | 0.05 GB |
| `adfc2_swat_sequential_n3` | 24 | 0.07 GB |
| `adfc2_swat_sequential_n3_oldmask` | 24 | 0.07 GB |
| `adfc2_swat_sequential_n5` | 36 | 0.11 GB |
| `adfc2_swat_sequential_n5_oldmask` | 36 | 0.11 GB |
| `adfc2_swat_window_W1` | 15 | 0.05 GB |
| `adfc2_swat_window_W1_oldmask` | 15 | 0.05 GB |
| `adfc2_swat_window_W2` | 15 | 0.05 GB |
| `adfc2_swat_window_W2_oldmask` | 15 | 0.05 GB |
| `adfc2_swat_window_W3` | 15 | 0.05 GB |
| `adfc2_swat_window_W3_oldmask` | 15 | 0.05 GB |
| `aeft_psm_opcm_scale` | 27 | 0.08 GB |
| `aeft_psm_sum_became` | 27 | 0.08 GB |
| `aeft_psm_sum_scale` | 27 | 0.08 GB |
| `basefrac_etth1_03` | 27 | 0.08 GB |
| `basefrac_etth1_07` | 27 | 0.08 GB |
| `basefrac_ettm2_03` | 27 | 0.08 GB |
| `basefrac_ettm2_07` | 27 | 0.08 GB |
| `basefrac_exchange_03` | 27 | 0.08 GB |
| `basefrac_exchange_07` | 27 | 0.08 GB |
| `continual_etth` | 24 | 0.07 GB |
| `continual_psm` | 24 | 0.06 GB |
| `continual_swat` | 24 | 0.18 GB |
| `etth2_continual_n2` | 18 | 0.05 GB |
| `etth2_continual_n3` | 24 | 0.07 GB |
| `etth2_continual_n5` | 36 | 0.10 GB |
| `etth2_gate_base` | 6 | 0.02 GB |
| `etth2_gate_joint` | 6 | 0.02 GB |
| `etth2_merge_n2` | 21 | 0.06 GB |
| `etth2_merge_n3` | 27 | 0.08 GB |
| `etth2_merge_n5` | 39 | 0.11 GB |
| `etth2_window_W1` | 15 | 0.04 GB |
| `etth2_window_W2` | 15 | 0.04 GB |
| `etth2_window_W3` | 15 | 0.04 GB |
| `ettm2_continual_n2` | 18 | 0.05 GB |
| `ettm2_continual_n3` | 24 | 0.07 GB |
| `ettm2_continual_n5` | 36 | 0.10 GB |
| `ettm2_gate_base` | 6 | 0.02 GB |
| `ettm2_gate_joint` | 6 | 0.02 GB |
| `ettm2_merge_n2` | 21 | 0.06 GB |
| `ettm2_merge_n3` | 27 | 0.08 GB |
| `ettm2_merge_n5` | 39 | 0.11 GB |
| `ettm2_window_W1` | 15 | 0.04 GB |
| `ettm2_window_W2` | 15 | 0.04 GB |
| `ettm2_window_W3` | 15 | 0.04 GB |
| `exch_continual` | 24 | 0.07 GB |
| `exch_gate_standard` | 6 | 0.02 GB |
| `exch_incremental` | 27 | 0.08 GB |
| `n1_etth1` | 15 | 0.04 GB |
| `n1_exchange` | 15 | 0.04 GB |
| `n1_psm` | 15 | 0.04 GB |
| `n1_swat` | 15 | 0.12 GB |
| `noisefloor_etth` | 27 | 0.08 GB |
| `noisefloor_psm` | 27 | 0.07 GB |
| `noisefloor_std_etth` | 6 | 0.02 GB |
| `noisefloor_std_psm` | 6 | 0.02 GB |
| `noisefloor_std_swat` | 6 | 0.05 GB |
| `noisefloor_swat` | 27 | 0.21 GB |
| `opcm2_psm_opcm_became` | 27 | 0.08 GB |
| `opcm2_psm_opcm_became_n2` | 21 | 0.06 GB |
| `opcm2_psm_opcm_became_n5` | 39 | 0.11 GB |
| `opcm2_psm_opcm_scale` | 27 | 0.08 GB |
| `opcm2_psm_opcm_scale_n2` | 21 | 0.06 GB |
| `opcm2_psm_opcm_scale_n5` | 39 | 0.11 GB |
| `opcm2_psm_sum_became` | 27 | 0.08 GB |
| `opcm2_psm_sum_became_n2` | 21 | 0.06 GB |
| `opcm2_psm_sum_became_n5` | 39 | 0.11 GB |
| `opcm2_psm_sum_scale` | 27 | 0.08 GB |
| `opcm2_psm_sum_scale_n2` | 21 | 0.06 GB |
| `opcm2_psm_sum_scale_n5` | 39 | 0.11 GB |
| `opcm2_psm_thr03` | 27 | 0.08 GB |
| `opcm2_psm_thr04` | 27 | 0.08 GB |
| `opcm2_psm_thr06` | 27 | 0.08 GB |
| `opcm2_psm_thr07` | 27 | 0.08 GB |
| `origin_etth1_joint_f075` | 6 | 0.02 GB |
| `origin_etth1_joint_f0875` | 6 | 0.02 GB |
| `origin_etth1_joint_f10` | 6 | 0.02 GB |
| `origin_etth1_merge_f075` | 27 | 0.08 GB |
| `origin_etth1_merge_f0875` | 27 | 0.08 GB |
| `origin_etth1_merge_f10` | 27 | 0.08 GB |
| `origin_etth1_sequential_f075` | 24 | 0.07 GB |
| `origin_etth1_sequential_f0875` | 24 | 0.07 GB |
| `origin_etth1_sequential_f10` | 24 | 0.07 GB |
| `origin_etth1_window_f075` | 15 | 0.04 GB |
| `origin_etth1_window_f0875` | 15 | 0.04 GB |
| `origin_etth1_window_f10` | 15 | 0.04 GB |
| `origin_etth2_joint_f075` | 6 | 0.02 GB |
| `origin_etth2_joint_f0875` | 6 | 0.02 GB |
| `origin_etth2_merge_f075` | 27 | 0.08 GB |
| `origin_etth2_merge_f0875` | 27 | 0.08 GB |
| `origin_etth2_sequential_f075` | 24 | 0.07 GB |
| `origin_etth2_sequential_f0875` | 24 | 0.07 GB |
| `origin_etth2_window_f075` | 15 | 0.04 GB |
| `origin_etth2_window_f0875` | 15 | 0.04 GB |
| `origin_ettm2_joint_f075` | 6 | 0.02 GB |
| `origin_ettm2_joint_f0875` | 6 | 0.02 GB |
| `origin_ettm2_merge_f075` | 27 | 0.08 GB |
| `origin_ettm2_merge_f0875` | 27 | 0.08 GB |
| `origin_ettm2_sequential_f075` | 24 | 0.07 GB |
| `origin_ettm2_sequential_f0875` | 24 | 0.07 GB |
| `origin_ettm2_window_f075` | 15 | 0.04 GB |
| `origin_ettm2_window_f0875` | 15 | 0.04 GB |
| `origin_exchange_joint_f075` | 6 | 0.02 GB |
| `origin_exchange_joint_f0875` | 6 | 0.02 GB |
| `origin_exchange_joint_f10` | 6 | 0.02 GB |
| `origin_exchange_merge_f075` | 27 | 0.08 GB |
| `origin_exchange_merge_f0875` | 27 | 0.08 GB |
| `origin_exchange_merge_f10` | 27 | 0.08 GB |
| `origin_exchange_sequential_f075` | 24 | 0.07 GB |
| `origin_exchange_sequential_f0875` | 24 | 0.07 GB |
| `origin_exchange_sequential_f10` | 24 | 0.07 GB |
| `origin_exchange_window_f075` | 15 | 0.04 GB |
| `origin_exchange_window_f0875` | 15 | 0.04 GB |
| `origin_exchange_window_f10` | 15 | 0.04 GB |
| `segsweep_etth1_merge_n2` | 21 | 0.06 GB |
| `segsweep_etth1_merge_n5` | 39 | 0.11 GB |
| `segsweep_etth1_seq_n2` | 18 | 0.05 GB |
| `segsweep_etth1_seq_n5` | 36 | 0.10 GB |
| `segsweep_exchange_merge_n2` | 21 | 0.06 GB |
| `segsweep_exchange_merge_n5` | 39 | 0.11 GB |
| `segsweep_exchange_seq_n2` | 18 | 0.05 GB |
| `segsweep_exchange_seq_n5` | 36 | 0.10 GB |
| `segsweep_psm_merge_n2` | 21 | 0.05 GB |
| `segsweep_psm_merge_n5` | 39 | 0.10 GB |
| `segsweep_psm_seq_n2` | 18 | 0.05 GB |
| `segsweep_psm_seq_n5` | 36 | 0.09 GB |
| `segsweep_swat_merge_n2` | 21 | 0.16 GB |
| `segsweep_swat_merge_n5` | 39 | 0.30 GB |
| `segsweep_swat_seq_n2` | 18 | 0.14 GB |
| `segsweep_swat_seq_n5` | 36 | 0.28 GB |
| `selalpha_etth1_n3` | 27 | 0.08 GB |
| `selalpha_exchange_n3` | 27 | 0.08 GB |
| `window_etth1_W1` | 15 | 0.04 GB |
| `window_etth1_W2` | 15 | 0.04 GB |
| `window_etth1_W3` | 15 | 0.04 GB |
| `window_exchange_W1` | 15 | 0.04 GB |
| `window_exchange_W2` | 15 | 0.04 GB |
| `window_exchange_W3` | 15 | 0.04 GB |
| `window_psm_W1` | 15 | 0.04 GB |
| `window_psm_W2` | 15 | 0.04 GB |
| `window_psm_W3` | 15 | 0.04 GB |
| `window_swat_W1` | 15 | 0.12 GB |
| `window_swat_W2` | 15 | 0.12 GB |
| `window_swat_W3` | 15 | 0.12 GB |

Per-file sizes and SHA-256 are in `results_archive/checkpoints.csv` (3921 rows) — kept as CSV rather than inlined here because a 3921-row table in Markdown is not readable and not greppable.
