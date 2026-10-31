# First-use maths and implementation coverage

Each concept is introduced where it becomes necessary. This map is author review evidence, not a learner assessment.

| Concept | First lesson | Background taught there | Worked calculation | Check / boundary |
| --- | --- | --- | --- | --- |
| Evidence conjunction | 01 | Boolean conjunction and sets | Correct number/wrong passage | 01_contract.py; extra-claim transfer |
| Source/entry-point lifecycle | 02 | Imports, streams, exit status, packaged resources | Four actual actions from /tmp | 02_scaffold_trace.py live mode |
| Token capacity | 03 | Integer addition and identity | 2048+256 ≤2304 | 03_compatibility.py negative capability |
| Storage and live budgets | 04 | Bits/bytes, powers of two, ceiling, overlapping lifetimes | 8B at 4-bit + per-32 scale | 04_memory_budget.py; four interactive cases |
| Canonical spans | 05 | UTF-8 vs Unicode indices, hash identity, half-open intervals | [0,28), [30,46) | 05_passages.py newline/edit cases |
| Data roles and packing | 06 | Groups, disjoint sets, modular hash, bounded sums | 3+4 fits; 3+4+2 fails | 06_splits.py complete prompt policy departure |
| Softmax and decode | 07 | Exponent/log, normalization, argmax, common factor cancellation | [0,log3] → [0.25,0.75] | 07_reference_decode.py common-shift/EOS/length |
| Quantizer grid | 08 | Range division, ties-to-even, saturation, step error | [-2,4], 3-bit, s=6/7,z=2 | 08_scalar_quant.py exact endpoint and range audit |
| Integer dot scale | 08 | Distributive multiplication and summed products | [2,-1] dot [3,4] ×0.5×0.25 =0.25 | Visible derivation; W/A/accumulator distinction |
| Packed representation | 09 | Bits, masks, shifts, sign extension, odd count | [-8,-1,0,7,3] → [248,112,3] | C++20 roundtrip; independent Python nibble audit |
| Linear error and norm | 10 | Rows/columns, dot product, subtraction, square root, Cauchy-Schwarz | E x =[-0.98,-0.95] | 10_layer_error.py and rational changed-input audit |
| Gram reconstruction | 12 | Uncentered second moments, expansion of finite sums | G=diag(16,1), losses1 vs0.0625 | 12_calibration.py direct-vs-quadratic and duplication check |
| Equivalent diagonal transform | 13 | Diagonal inverse, exact identity vs nonlinear quantizer | Scale4 preserves1.98, ternary MSE0.0036 | 13_activation_scaling.py grid and float identity |
| Constrained quadratic | 14 | Partial slope, derivative of square, positive curvature, matrix inverse | e1=.25,e2=-.25 halves loss | 14_compensation.py brute grid and finite difference |
| Precision allocation | 15 | Ceil per group, finite Cartesian product, constrained minimum | 900-byte cap chooses848 | 15_precision_budget.py; 700-byte transfer672 |
| Task rates and pairs | 16 | Indicators, denominators, set alignment, mean differences | Support1/2, false refusal1/2, unsupported1/3 | 16_quality.py; missing-ID rejection contract |
| Timing | 17 | Event subtraction, work/interval, nearest-rank quantile | Five tokens/four intervals →2tok/s | 17_timing.py; explicit invented timestamps |
| Cache allocation | 18 | Tensor axes/products, GQA, owner dedup, min(window,T) | T2048 →2359296 bytes | 18_cache_budget.py; not Gemma map |
| Attention/cache error | 19 | Dot scores, softmax derivative, weighted values, residual sum | Base3; key3.6; value3.75 | 19_cache_error.py; Jacobian numerical audit |
| Artifact persistence | 20 | Hash/size, immutable identity, rename visibility vs durability | Corrupt2 +partial3 →verified1 | 20_recovery.py and new-process toy reload |
| Service state | 21–22 | Request ownership, try/finally, schemas, HTTP lifecycle | Invalid citation still resets; HTTP200/400 | 21_answer_boundary.py;22_local_app.py fixture-only |
| Decision uncertainty | 23 | Conjunctive feasibility, paired differences, replacement resampling | No feasible winner;4**4=256 | 23_selection.py; full range not claimed CI |
