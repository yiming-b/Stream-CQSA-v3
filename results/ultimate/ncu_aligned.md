# M0.3: ncu stall attribution of the all-live tax on the aligned layout

Task: c=7 itr=1 path=[2] N=131072 L=56064 (L%128=0), aligned_128 layout, gathered inputs, H=8 D=64 fp16 causal. One kernel launch per arm, one process per arm, `ncu --set full --section SourceCounters --section WarpStateStats --clock-control base`.

GPU: NVIDIA A100 80GB PCIe (NVIDIA A100 80GB PCIe, 210 MHz, 1410 MHz), node della-l02g14, slurm job 14292338, torch 2.10.0+cu130, module cqsa_cuda_next_v11. PCIe node: relative metrics only, never compare durations with SXM timings.

FA-2 arm = torch SDPA FLASH_ATTENTION backend (flash_attn not installed); SDPA-flash is ~5% slower than the flash_attn package. Arms: plain = v11 module with CQS off (bits=None, upstream loop); zero = CQS on, all-zero bits (all tiles live); real = CQS on, task bits (22% masked tiles skipped).


## Per-arm metrics

| metric | fa2 | plain | zero | real |
|---|---|---|---|---|
| kernel | void flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, 0, 0, half_t, Flash_kernel_... | void flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, 0, 0, half_t, Flash_kernel_... | void flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, 0, 0, half_t, Flash_kernel_... | void flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, 0, 0, half_t, Flash_kernel_... |
| duration ms (PCIe, base clock) | 23.58 | 22.79 | 29.75 | 23.58 |
| SM clock GHz | 1.06 | 1.06 | 1.06 | 1.06 |
| elapsed cycles | 25,112,463 | 24,267,513 | 31,686,011 | 25,113,171 |
| executed instructions | 5,060,357,568 | 3,662,927,328 | 4,131,040,480 | 3,285,508,000 |
| issued instructions | 5,060,384,783 | 3,662,957,510 | 4,134,139,128 | 3,288,606,644 |
| warp cycles / issued inst | 4.03 | 5.35 | 6.23 | 6.26 |
| eligible warps / scheduler | 0.66 | 0.48 | 0.37 | 0.37 |
| active warps / scheduler | 1.97 | 1.96 | 1.96 | 1.96 |
| issued / scheduler / cycle | 0.49 | 0.37 | 0.32 | 0.31 |
| issue slots busy % | 49.02 | 36.63 | 31.55 | 31.33 |
| tensor pipe active % | 61.04 | 63.03 | 48.08 | 46.77 |
| SM throughput % | 58.26 | 60.28 | 46.12 | 45.40 |
| registers / thread | 255.00 | 255.00 | 255.00 | 255.00 |
| stack B | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 |
| dyn smem / block B | 49,152.00 | 50,176.00 | 50,176.00 | 50,176.00 |
| grid | 3,504.00 | 3,504.00 | 3,504.00 | 3,504.00 |
| waves / SM | 16.22 | 16.22 | 16.22 | 16.22 |
| occupancy theoretical % | 12.50 | 12.50 | 12.50 | 12.50 |
| occupancy achieved % | 12.34 | 12.26 | 12.28 | 12.26 |
| L1 hit % | 26.35 | 55.47 | 59.59 | 59.57 |
| L2 hit % | 98.76 | 93.29 | 94.14 | 98.34 |
| DRAM throughput % | 0.60 | 3.48 | 2.40 | 1.08 |
| global load bytes | n/a | n/a | n/a | n/a |
| global load requests | 49,336,320 | 49,336,320 | 78,482,592 | 64,218,400 |
| local load bytes | n/a | n/a | n/a | n/a |
| local store bytes | n/a | n/a | n/a | n/a |
| branch targets uniform % | 99.85 | 99.89 | 99.97 | 99.97 |
| not-pred-off threads / warp | 32.00 | 32.00 | 32.00 | 32.00 |

## Stall breakdown (warp cycles per issued instruction, by reason; sums to the row above)

| reason | fa2 cyc (% of total) | plain cyc (% of total) | zero cyc (% of total) | real cyc (% of total) |
|---|---|---|---|---|
| wait | 1.36 (33.7%) | 2.33 (43.6%) | 2.84 (45.6%) | 2.84 (45.4%) |
| selected | 1.00 (24.8%) | 1.00 (18.7%) | 1.00 (16.1%) | 1.00 (16.0%) |
| math_pipe_throttle | 0.77 (19.1%) | 0.82 (15.3%) | 0.40 (6.4%) | 0.38 (6.1%) |
| short_scoreboard | 0.15 (3.7%) | 0.17 (3.2%) | 0.64 (10.3%) | 0.62 (9.9%) |
| long_scoreboard | 0.08 (2.0%) | 0.07 (1.3%) | 0.50 (8.0%) | 0.56 (8.9%) |
| not_selected | 0.36 (8.9%) | 0.30 (5.6%) | 0.17 (2.7%) | 0.17 (2.7%) |
| no_instruction | 0.03 (0.7%) | 0.04 (0.7%) | 0.21 (3.4%) | 0.23 (3.7%) |
| barrier | 0.08 (2.0%) | 0.22 (4.1%) | 0.13 (2.1%) | 0.12 (1.9%) |
| mio_throttle | 0.03 (0.7%) | 0.21 (3.9%) | 0.14 (2.2%) | 0.13 (2.1%) |
| dispatch_stall | 0.17 (4.2%) | 0.14 (2.6%) | 0.12 (1.9%) | 0.12 (1.9%) |
| branch_resolving | 0.01 (0.2%) | 0.01 (0.2%) | 0.08 (1.3%) | 0.08 (1.3%) |
| lg_throttle | 0.00 (0.0%) | 0.04 (0.7%) | 0.01 (0.2%) | 0.01 (0.2%) |
| drain | 0.00 (0.0%) | 0.00 (0.0%) | 0.00 (0.0%) | 0.00 (0.0%) |
| misc | 0.00 (0.0%) | 0.00 (0.0%) | 0.00 (0.0%) | 0.00 (0.0%) |
| imc_miss | 0.00 (0.0%) | 0.00 (0.0%) | 0.00 (0.0%) | 0.00 (0.0%) |
| membar | 0.00 (0.0%) | 0.00 (0.0%) | 0.00 (0.0%) | 0.00 (0.0%) |
| sleeping | 0.00 (0.0%) | 0.00 (0.0%) | 0.00 (0.0%) | 0.00 (0.0%) |
| tex_throttle | 0.00 (0.0%) | 0.00 (0.0%) | 0.00 (0.0%) | 0.00 (0.0%) |

## Pipe utilisation (% of peak sustained active, top pipes)

| pipe | fa2 | plain | zero | real |
|---|---|---|---|---|
| adu | 0.48 | 0.50 | 0.38 | 0.37 |
| alu | 23.28 | 16.81 | 13.39 | 13.67 |
| cbu | 0.00 | 0.00 | 0.48 | 0.52 |
| fma | 43.73 | 20.87 | 20.37 | 20.00 |
| lsu | 14.82 | 22.44 | 18.43 | 18.16 |
| tensor_op_hmma | 15.26 | 15.76 | 12.02 | 11.69 |
| uniform | 0.06 | 0.01 | 0.05 | 0.05 |
| xu | 31.48 | 32.51 | 24.80 | 24.12 |

## SASS stall samples by opcode class (SourceCounters, no -lineinfo: SASS view only)

Columns: % of all warp-stall samples in the kernel, then executed warp-instructions of that class in % of the kernel's total.

| class | fa2 samples% / inst% | plain samples% / inst% | zero samples% / inst% | real samples% / inst% |
|---|---|---|---|---|
| mma | 42.6 / 15.6 | 44.4 / 21.5 | 34.7 / 19.1 | 33.6 / 18.7 |
| fp32 | 37.7 / 60.8 | 19.2 / 40.0 | 20.9 / 35.8 | 20.2 / 35.1 |
| int | 3.7 / 3.6 | 5.1 / 5.8 | 13.3 / 12.5 | 16.2 / 13.4 |
| exp_mufu | 4.9 / 8.0 | 15.2 / 11.1 | 10.1 / 9.8 | 9.8 / 9.6 |
| branch | 0.9 / 0.2 | 4.1 / 0.3 | 5.7 / 2.4 | 5.1 / 2.8 |
| local | 3.0 / 1.2 | 3.5 / 6.5 | 5.7 / 5.8 | 5.4 / 5.7 |
| smem_ld | 3.1 / 4.7 | 3.6 / 6.6 | 3.5 / 5.8 | 3.4 / 5.7 |
| convert | 2.6 / 4.5 | 3.3 / 6.2 | 3.4 / 6.1 | 3.3 / 6.0 |
| other | 0.0 / 0.0 | 0.0 / 0.0 | 1.5 / 1.1 | 1.8 / 1.3 |
| cp_async | 1.3 / 1.2 | 1.6 / 1.7 | 1.2 / 1.5 | 1.2 / 1.5 |
| barrier | 0.2 / 0.1 | 0.1 / 0.2 | 0.0 / 0.1 | 0.0 / 0.1 |
| nop | 0.0 / 0.0 | 0.0 / 0.0 | 0.1 / 0.0 | 0.1 / 0.0 |
| smem_st | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| global_st | 0.0 / 0.0 | 0.0 / 0.0 | n/a | n/a |
| global_ld | 0.0 / 0.0 | 0.0 / 0.0 | n/a | n/a |
| static SASS instructions | 4400 | 4032 | 72776 | 72776 |
| total stall samples | 1,259,932 | 1,219,586 | 1,598,673 | 1,278,362 |

## Top 10 SASS instructions by stall samples: fa2

| address | SASS | samples (% of all) | not-issued samples | executed | top reasons |
|---|---|---|---|---|---|
| 0x100052b09c0 | `      HMMA.16816.F32 R4, R200, R204, R4` | 20,763 (1.6%) | 14,145 | 3,062,496 | stall_math=13,903, stall_math (Not Issued)=10,929, stall_short_sb=3,346 |
| 0x100052b0760 | `      HMMA.16816.F32 R4, R196, R200, RZ` | 19,202 (1.5%) | 15,338 | 3,062,496 | stall_long_sb=15,162, stall_long_sb (Not Issued)=12,505, stall_short_sb=2,948 |
| 0x100052b0b00 | `      HMMA.16816.F32 R68, R200, R204, R68` | 16,370 (1.3%) | 7,499 | 3,062,496 | stall_math=11,523, stall_math (Not Issued)=6,181, stall_not_selected=2,322 |
| 0x100052b04a0 | `      LDL R5, [R1+0x4]` | 15,547 (1.2%) | 10,789 | 3,062,496 | stall_barrier=15,190, stall_barrier (Not Issued)=10,744, stall_selected=177 |
| 0x100052b4170 | `      HMMA.16816.F32 R52, R84, R80, R52` | 12,426 (1.0%) | 5,129 | 3,062,496 | stall_math=9,649, stall_math (Not Issued)=5,030, stall_not_selected=2,140 |
| 0x100052b0ba0 | `      HMMA.16816.F32 R100, R200, R204, R100` | 12,080 (1.0%) | 6,452 | 3,062,496 | stall_math=8,901, stall_math (Not Issued)=5,328, stall_wait=1,389 |
| 0x100052b0a60 | `      HMMA.16816.F32 R36, R200, R204, R36` | 11,967 (0.9%) | 5,802 | 3,062,496 | stall_math=8,838, stall_math (Not Issued)=4,804, stall_not_selected=1,499 |
| 0x100052b40d0 | `      FSETP.GEU.AND P6, PT, R45, -126, PT` | 10,445 (0.8%) | 3,661 | 3,062,496 | stall_math=5,049, stall_not_selected=4,894, stall_math (Not Issued)=3,477 |
| 0x100052b0f00 | `      HMMA.16816.F32 R4, R196, R208, R4` | 9,679 (0.8%) | 7,398 | 3,062,496 | stall_short_sb=5,824, stall_short_sb (Not Issued)=4,522, stall_math=3,109 |
| 0x100052b4550 | `@!P1  FMUL R189, R189, 0.5` | 8,834 (0.7%) | 2,483 | 3,062,496 | stall_not_selected=3,871, stall_math=3,795, stall_math (Not Issued)=1,975 |

## Top 10 SASS instructions by stall samples: plain

| address | SASS | samples (% of all) | not-issued samples | executed | top reasons |
|---|---|---|---|---|---|
| 0x100052b0a50 | `@!P0  BRA 0x100052b0ce0` | 31,013 (2.5%) | 21,280 | 3,062,496 | stall_barrier=30,725, stall_barrier (Not Issued)=21,265, stall_selected=189 |
| 0x100052b1050 | `      HMMA.16816.F32 R52, R128, R64, RZ` | 18,261 (1.5%) | 13,844 | 3,062,496 | stall_short_sb=11,483, stall_short_sb (Not Issued)=8,897, stall_math=4,286 |
| 0x100052b1950 | `@!P1  BRA 0x100052b1e50` | 17,671 (1.4%) | 12,809 | 3,062,496 | stall_barrier=17,429, stall_barrier (Not Issued)=12,808, stall_selected=187 |
| 0x100052b14e0 | `      HMMA.16816.F32 R36, R204, R212, R36` | 14,615 (1.2%) | 11,794 | 3,062,496 | stall_math=10,251, stall_math (Not Issued)=9,595, stall_not_selected=1,551 |
| 0x100052b16e0 | `      HMMA.16816.F32 R4, R212, R216, R4` | 14,287 (1.2%) | 10,480 | 3,062,496 | stall_math=8,975, stall_math (Not Issued)=7,831, stall_short_sb=3,568 |
| 0x100052b1450 | `      HMMA.16816.F32 R8, R216, R212, R8` | 12,106 (1.0%) | 8,886 | 3,062,496 | stall_math=7,406, stall_math (Not Issued)=6,404, stall_short_sb=1,832 |
| 0x100052b0f00 | `      HMMA.16816.F32 R4, R128, R16, RZ` | 11,487 (0.9%) | 8,366 | 3,062,496 | stall_long_sb=5,384, stall_math=4,272, stall_long_sb (Not Issued)=4,065 |
| 0x100052b1430 | `      HMMA.16816.F32 R4, R204, R212, R4` | 9,767 (0.8%) | 7,627 | 3,062,496 | stall_math=6,522, stall_math (Not Issued)=5,946, stall_wait=1,216 |
| 0x100052b5490 | `      HMMA.16816.F32 R176, R196, R184, R40` | 7,648 (0.6%) | 6,075 | 3,062,496 | stall_math=6,469, stall_math (Not Issued)=5,897, stall_not_selected=779 |
| 0x100052b55a0 | `      HMMA.16816.F32 R188, R200, R4, R52` | 7,504 (0.6%) | 5,751 | 3,062,496 | stall_math=6,212, stall_math (Not Issued)=5,631, stall_not_selected=725 |

## Top 10 SASS instructions by stall samples: zero

| address | SASS | samples (% of all) | not-issued samples | executed | top reasons |
|---|---|---|---|---|---|
| 0x1000533b140 | `@!P0  LOP3.LUT R52, R28, R30, RZ, 0xc0, !PT` | 18,335 (1.1%) | 13,933 | 3,048,512 | stall_long_sb=17,852, stall_long_sb (Not Issued)=13,788, stall_selected=190 |
| 0x10005339bd0 | `      LDL.64 R36, [R1+0xc0]` | 18,056 (1.1%) | 14,066 | 3,062,496 | stall_barrier=16,826, stall_barrier (Not Issued)=13,380, stall_wait=937 |
| 0x1000533c440 | `@!P0  BRA 0x1000533c9e0` | 16,379 (1.0%) | 12,477 | 3,062,496 | stall_barrier=15,178, stall_barrier (Not Issued)=11,774, stall_wait=970 |
| 0x1000533b050 | `@!P0  LOP3.LUT R16, R28, R30, RZ, 0xfc, !PT` | 15,793 (1.0%) | 12,103 | 3,048,512 | stall_long_sb=15,262, stall_long_sb (Not Issued)=11,940, stall_selected=214 |
| 0x1000533a2f0 | `      HMMA.16816.F32 R96, R112, R40, R96` | 12,784 (0.8%) | 11,086 | 3,062,496 | stall_short_sb=12,146, stall_short_sb (Not Issued)=10,773, stall_math=387 |
| 0x1000533acd0 | `      HMMA.16816.F32 R144, R40, R28, R144` | 9,748 (0.6%) | 7,491 | 3,062,496 | stall_short_sb=5,246, stall_short_sb (Not Issued)=3,976, stall_math=2,822 |
| 0x100053bffd0 | `      HMMA.16816.F32 R212, R52, R56, R212` | 9,181 (0.6%) | 7,089 | 3,062,496 | stall_short_sb=5,502, stall_short_sb (Not Issued)=4,473, stall_math=2,626 |
| 0x1000533ad30 | `      HMMA.16816.F32 R156, R32, R28, R156` | 8,919 (0.6%) | 6,523 | 3,062,496 | stall_short_sb=4,977, stall_short_sb (Not Issued)=3,667, stall_math=2,038 |
| 0x1000533a650 | `      HMMA.16816.F32 R164, R32, R44, R164` | 8,408 (0.5%) | 6,122 | 3,062,496 | stall_short_sb=7,060, stall_short_sb (Not Issued)=5,314, stall_math=958 |
| 0x1000533ab80 | `      HMMA.16816.F32 R104, R40, R28, R104` | 8,138 (0.5%) | 6,254 | 3,062,496 | stall_short_sb=5,197, stall_short_sb (Not Issued)=4,084, stall_math=1,303 |

## Top 10 SASS instructions by stall samples: real

| address | SASS | samples (% of all) | not-issued samples | executed | top reasons |
|---|---|---|---|---|---|
| 0x10005338bd0 | `      LDL.64 R36, [R1+0xc0]` | 13,274 (1.0%) | 10,418 | 2,385,056 | stall_barrier=12,366, stall_barrier (Not Issued)=9,883, stall_wait=719 |
| 0x1000533a140 | `@!P0  LOP3.LUT R52, R28, R30, RZ, 0xc0, !PT` | 12,983 (1.0%) | 9,885 | 2,371,072 | stall_long_sb=12,604, stall_long_sb (Not Issued)=9,762, stall_wait=157 |
| 0x1000533b440 | `@!P0  BRA 0x1000533b9e0` | 12,462 (1.0%) | 9,601 | 2,385,056 | stall_barrier=11,528, stall_barrier (Not Issued)=9,067, stall_wait=719 |
| 0x1000533a050 | `@!P0  LOP3.LUT R16, R28, R30, RZ, 0xfc, !PT` | 11,132 (0.9%) | 8,490 | 2,371,072 | stall_long_sb=10,721, stall_long_sb (Not Issued)=8,369, stall_selected=170 |
| 0x100053392f0 | `      HMMA.16816.F32 R96, R112, R40, R96` | 9,868 (0.8%) | 8,579 | 2,385,056 | stall_short_sb=9,374, stall_short_sb (Not Issued)=8,326, stall_math=294 |
| 0x10005339cd0 | `      HMMA.16816.F32 R144, R40, R28, R144` | 7,519 (0.6%) | 5,758 | 2,385,056 | stall_short_sb=4,190, stall_short_sb (Not Issued)=3,203, stall_math=1,968 |
| 0x100053befd0 | `      HMMA.16816.F32 R212, R52, R56, R212` | 7,046 (0.6%) | 5,382 | 2,385,056 | stall_short_sb=4,212, stall_short_sb (Not Issued)=3,410, stall_math=1,969 |
| 0x10005339d30 | `      HMMA.16816.F32 R156, R32, R28, R156` | 6,725 (0.5%) | 4,941 | 2,385,056 | stall_short_sb=3,755, stall_short_sb (Not Issued)=2,771, stall_math=1,521 |
| 0x10005339650 | `      HMMA.16816.F32 R164, R32, R44, R164` | 6,334 (0.5%) | 4,605 | 2,385,056 | stall_short_sb=5,383, stall_short_sb (Not Issued)=4,050, stall_math=672 |
| 0x10005339b80 | `      HMMA.16816.F32 R104, R40, R28, R104` | 6,268 (0.5%) | 4,764 | 2,385,056 | stall_short_sb=4,008, stall_short_sb (Not Issued)=3,155, stall_wait=1,039 |

## Verdict

(written by hand after reading the tables; see the bottom of this file)

