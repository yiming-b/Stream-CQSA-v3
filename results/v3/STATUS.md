# results/v3 status (2026-10-31 23:34)

`/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/v3/results.jsonl`: 550 distinct configurations recorded (last row per key).

| experiment | part | planned | ok | oom | error | missing | jobs |
|---|---|---|---|---|---|---|---|
| E0 | all | 126 | 126 | 0 | 0 | 0 | 15966097 |
| E0b | all | 14 | 14 | 0 | 0 | 0 | 16328779 |
| E1 | fwd | 36 | 36 | 0 | 0 | 0 | 15966097, 16003186 |
| E1 | bwd | 36 | 36 | 0 | 0 | 0 | 16005563, 16007771, 16011674, 16056869, 16092420 |
| E2 | fwd | 24 | 24 | 0 | 0 | 0 | 16072865, 16074280, 16074929 |
| E2 | bwd | 24 | 24 | 0 | 0 | 0 | 16074929, 16075878, 16113653, 16115186, 16115960 |
| E2 | pcie_fwd | 12 | 12 | 0 | 0 | 0 | 16057173, 16061811, 16077768, 16079116, 16079918 |
| E2 | pcie_bwd | 0 | 0 | 0 | 0 | 0 | - |
| E3 | fwd | 36 | 36 | 0 | 0 | 0 | 15966097, 16003186 |
| E3 | bwd | 36 | 36 | 0 | 0 | 0 | 16007771, 16011674, 16059599, 16061624, 16063609, 16070367 |
| E4 | all | 32 | 32 | 0 | 0 | 0 | 16003186, 16005563, 16116751 |
| E5 | all | 20 | 20 | 0 | 0 | 0 | 16070367, 16072865 |
| E6 | all | 44 | 44 | 0 | 0 | 0 | 16005563, 16007771, 16011674, 16092420, 16094767 |
| E9 | all | 42 | 42 | 0 | 0 | 0 | 16333022, 16334737, 16500372, 16500373 |
| E10 | fwd | 18 | 18 | 0 | 0 | 0 | 16559720 |
| E10 | bwd | 18 | 18 | 0 | 0 | 0 | 16559720, 16562804 |
| E11 | all | 4 | 4 | 0 | 0 | 0 | 16559720 |
| E12 | all | 10 | 10 | 0 | 0 | 0 | 16559720 |
| E8 | all | 16 | 16 | 0 | 0 | 0 | 16116751, 16119471 |

**Missing rows: 0**

## Excluded from the plan (findings, not measurements)

- arm=classic_Triton/wave_Triton, D=128, direction=bwd: Triton backward at D=128 does not finish (known 128-wide key-tile pathology); CUDA backward is the default
- exp=E2, node=pcie, direction=bwd: not run: 1.5-3 h per row at 150 GB host RAM on gpu-short; the SXM 4M rows and the PCIe forward rows cover the question
