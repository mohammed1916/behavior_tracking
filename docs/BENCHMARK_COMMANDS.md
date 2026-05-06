# Benchmark Commands

This document lists the exact commands used to generate binary labels, validate them, and run the VLM benchmark in this workspace.

## Environment

Python executable used:

```powershell
& 'C:\ProgramData\anaconda3\python.exe'
```

For scripts that import `backend.*`, `PYTHONPATH` was set to the repo root:

```powershell
$env:PYTHONPATH='.'
```

## 1. Generate Binary Labels From Procedure Anchors

Converts the Assembly dataset's `procedure_anchors.csv` into binary frame labels using:
- inside procedure segments => `work`
- outside procedure segments => `idle`

```powershell
& 'C:\ProgramData\anaconda3\python.exe' scripts/parse_procedure_anchors.py 'dataset\sequential_human_assembly\Scenario-B-Assembly\Assembly\procedure_anchors.csv' --all --output 'dataset\sequential_human_assembly\Scenario-B-Assembly\Assembly\binary_labels' --mode binary
```

Single-video export used for retest:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' scripts/parse_procedure_anchors.py 'dataset\sequential_human_assembly\Scenario-B-Assembly\Assembly\procedure_anchors.csv' --video-id s2_p1_1_a_1 --output 'dataset\sequential_human_assembly\Scenario-B-Assembly\Assembly\binary_labels\s2_p1_1_a_1_retest.csv' --mode binary
```

## 2. Validate Generated Label Files

Folder-wide validation of generated `*_labels.csv` files:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -c "import csv, glob, os, statistics as st; paths=glob.glob(r'dataset/sequential_human_assembly/Scenario-B-Assembly/Assembly/binary_labels/*_labels.csv'); counts=[]; idle_files=0; bad=[]; 
for p in paths:
    rows=list(csv.DictReader(open(p, newline='', encoding='utf-8')))
    labels={r['label'] for r in rows}
    if not labels <= {'work','idle'}: bad.append((os.path.basename(p), sorted(labels)))
    idle=sum(1 for r in rows if r['label']=='idle')
    work=sum(1 for r in rows if r['label']=='work')
    counts.append((os.path.basename(p), len(rows), work, idle))
    if idle>0: idle_files += 1
print('files', len(paths))
print('idle_files', idle_files)
print('bad', bad[:5])
print('rows_min', min(c[1] for c in counts), 'rows_max', max(c[1] for c in counts), 'rows_median', int(st.median(c[1] for c in counts)))
print('sample', counts[:5])"
```

Spot check for one known gap in `s2_p1_1_a_1_labels.csv`:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -c "import csv; from collections import Counter; p=r'dataset/sequential_human_assembly/Scenario-B-Assembly/Assembly/binary_labels/s2_p1_1_a_1_labels.csv'; rows=list(csv.DictReader(open(p, newline='', encoding='utf-8'))); c=Counter(r['label'] for r in rows); print('rows', len(rows)); print(dict(c)); print('sample_gaps', [(r['frame_index'], r['label']) for r in rows[620:646]])"
```

## 3. Probe BLIP VLM On One Video

This was used to confirm that BLIP could caption a local Assembly video before running the benchmark:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' scripts/standalone_run.py --video 'dataset\sequential_human_assembly\Scenario-B-Assembly\Assembly\raw video\s2_p1_1_a_1.avi' --model 'Salesforce/blip-image-captioning-large'
```

## 4. Run BLIP Binary Benchmark

Benchmarks BLIP against the generated binary labels.

Command used:

```powershell
$env:PYTHONPATH='.'; & 'C:\ProgramData\anaconda3\python.exe' scripts/benchmark_vlm_binary.py --videos-dir 'dataset\sequential_human_assembly\Scenario-B-Assembly\Assembly\raw video' --labels-dir 'dataset\sequential_human_assembly\Scenario-B-Assembly\Assembly\binary_labels' --model 'Salesforce/blip-image-captioning-large' --sample-every 60 --max-videos 5 --output 'logs\vlm_binary_benchmark_2026-04-10.json'
```

Artifacts produced:
- `logs\vlm_binary_benchmark_2026-04-10.json`
- `logs\vlm_binary_benchmark_2026-04-10_summary.txt`

## 5. Probe Qwen VLM

This was used to test whether Qwen was runnable offline in the current environment:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' scripts/standalone_run.py --video 'dataset\sequential_human_assembly\Scenario-B-Assembly\Assembly\raw video\s2_p1_1_a_1.avi' --model 'Qwen/Qwen2-VL-2B-Instruct'
```

## 6. Attempt Qwen Binary Benchmark

This benchmark attempt failed because Qwen still required Hugging Face metadata access while offline mode was enabled.

```powershell
$env:PYTHONPATH='.'; & 'C:\ProgramData\anaconda3\python.exe' scripts/benchmark_vlm_binary.py --videos-dir 'dataset\sequential_human_assembly\Scenario-B-Assembly\Assembly\raw video' --labels-dir 'dataset\sequential_human_assembly\Scenario-B-Assembly\Assembly\binary_labels' --model 'Qwen/Qwen2-VL-2B-Instruct' --sample-every 120 --max-videos 1 --output 'logs\qwen_vlm_binary_benchmark_2026-04-11.json'
```

## 7. Run Aggregation Tests

This was used earlier while checking available repo-native metrics and test coverage:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m pytest backend/tests/aggregation -q
```

## 8. Run Detector Integration Test

This was used to validate the detector pipeline in the current environment:

```powershell
$env:PYTHONPATH='.'; & 'C:\ProgramData\anaconda3\python.exe' backend/tests/test_detectors_integration.py
```

## Notes

- The BLIP benchmark that completed successfully used only the first 5 videos because `--max-videos 5` was specified.
- The 5 evaluated videos were:
  - `s2_p10_1_a_1`
  - `s2_p10_1_a_2`
  - `s2_p10_1_a_3`
  - `s2_p10_2_a_1`
  - `s2_p10_2_a_2`
