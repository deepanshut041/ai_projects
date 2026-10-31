## 17. Give every latency number a start, an end and a workload

An artifact can preserve answers while changing how long the application takes. To measure that honestly, define the timing boundary before collecting results. “Tokens per second” can describe prompt processing, generated-token intervals, a batch aggregate or an entire request. Those numbers cannot be interchanged.

### Draw the request timeline

Let $t_0$ be user-request arrival, $t_s$ submission to the runtime, $t_1$ the first observable output token and $t_n$ the last observable output token. Application TTFT is $t_1-t_0$; runtime TTFT is $t_1-t_s$. They differ by parsing, queueing, prompt construction and other pre-runtime work. If the endpoint buffers output, the first received chunk may contain multiple tokens; call it a chunk-observation metric unless exact token events are available.

For $n\ge2$ generated tokens, steady decode throughput over observed token events is $(n-1)/(t_n-t_1)$. The first token is already present at the start of that interval. Using $n$ over the same interval overstates this particular rate. For one token there are no inter-token intervals, so this rate is undefined. Prefill throughput uses processed prompt tokens divided by a separately instrumented prefill interval; TTFT also includes first-token computation and runtime overhead, so it is not automatically that interval.

![Timing boundaries](figures/timing.svg)

Our invented timestamps are $t_0=0$, $t_s=0.2$, $t_1=0.7$, $t_n=2.7$, with five generated tokens. User TTFT is 0.7 seconds; runtime TTFT is 0.5; decode rate is $4/2=2$ tokens/s; end-to-end time is 2.7 seconds. If 100 prompt tokens took an independently measured 0.4 seconds, prefill rate is 250 tokens/s. These are arithmetic fixtures, not a performance result.

### Run the timestamp reference

Create `17_timing.py`. The summarizer validates event order and counts. `nearest_rank()` defines one particular sample percentile estimator: sort $m$ observations and take index $\lceil pm\rceil-1$. For five values `[1,2,3,4,9]`, the p95 estimate is 9. Many other estimators interpolate; name your choice and retain the observations rather than implying that a p95 from five runs is stable.

[Download the complete reference](examples/17_timing.py)

```python
"""Timestamp arithmetic only; the timestamps below are invented fixtures."""
import json
import math


def summarize(start, submitted, first, last, tokens, prompt_tokens, prefill_seconds):
    if not start <= submitted <= first <= last or tokens < 2 or prefill_seconds <= 0:
        raise ValueError("Invalid timestamp/count contract")
    return {"end_to_end_s": last-start, "runtime_ttft_s": first-submitted,
            "user_ttft_s": first-start, "decode_tokens_s": (tokens-1)/(last-first),
            "prefill_tokens_s": prompt_tokens/prefill_seconds}


def nearest_rank(values, probability):
    if not values or not 0 < probability <= 1:
        raise ValueError("Need samples and a probability in (0,1]")
    ordered = sorted(values)
    return ordered[math.ceil(probability*len(values))-1]


def main():
    result = summarize(0, .2, .7, 2.7, 5, 100, .4)
    assert result["decode_tokens_s"] == 2 and result["prefill_tokens_s"] == 250
    print(json.dumps({k: round(v, 6) for k, v in result.items()} |
                     {"p95_fixture_s": nearest_rank([1, 2, 3, 4, 9], .95), "measured": False}))


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 17_timing.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "end_to_end_s": 2.7,
  "runtime_ttft_s": 0.5,
  "user_ttft_s": 0.7,
  "decode_tokens_s": 2.0,
  "prefill_tokens_s": 250.0,
  "p95_fixture_s": 9,
  "measured": false
}
```

Use a monotonic high-resolution timer for actual durations. On asynchronous devices, a host call returning may mean work was queued rather than completed. Synchronize the measured backend operation or use verified device/runtime event instrumentation, and record the cost of that boundary. Measure an actual streamed completion if reporting user TTFT. Do not place a timer around a lazy iterator without consuming it.

### Cold, warm and repeated work

Cold model loading belongs in its own receipt: fresh process, artifact read/mapping, initialization and first request. An OS file cache may already be warm even in a new process, so “fresh process” is more precise than “cold disk” unless cache state is controlled. Warm inference retains model/runtime state but must reset request-specific KV state according to the protocol. Report warm-up count, repeated run count and failed requests.

Use the same prompts, input lengths, output policy, placement, thread settings, batch/concurrency and runtime build for a mechanism comparison. If one answer terminates after five tokens and another after 200, end-to-end latency includes different work. Report actual generated counts and finish reasons. A fixed-token microbenchmark can isolate throughput; natural stopping on the task measures user behavior. Both are useful when named separately.

Randomize or alternate candidate order within a declared bounded protocol to reduce thermal/background drift. Retain per-request memory/timing records, summarize medians and tails and disclose background allocations. Do not run unrelated GPU-heavy jobs simultaneously and interpret the difference as a quantization effect.

[[practice:timing-transfer]]

### Interpret what low-bit weights can and cannot promise

Less weight traffic can help a bandwidth-limited decode, while unpacking overhead or unsupported kernels can erase the benefit. A compute-heavy prefill may respond differently. We infer no speedup from payload size alone. Keep conversion/calibration cost separate from serving cost, and state how many requests would be needed to amortize an expensive preparation step if that matters.

We can now compare fixed-length weight variants. The next study holds weights fixed and changes context/cache behavior, because long prompts create a second memory problem even after weights fit.
