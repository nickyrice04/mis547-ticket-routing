# Model scaling study

## Accuracy against size

| Tier | Model | Params (M) | Accuracy | Macro-F1 | Train time | Device |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | TF-IDF 1-2gram + LogisticRegression | 0.61 | 61.6% | 0.625 | 0.1 min | cpu |
| 2 | distilbert-base-uncased | 67.0 | 39.2% | 0.411 | 13.3 min | mps |

## Cost to serve one ticket

| Tier | Disk (MB) | RAM held (MB) | p50 at 1 thread | p50 at 2 threads | p95 at 2 threads |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | 164 | 1.2 ms | 1.2 ms | 1.3 ms |

## Confidence threshold

How much of the queue each model can route on its own, and how often it is right.

| Tier | Threshold | Auto-routed | Accuracy when auto-routed | Sent to a person |
| --- | --- | --- | --- | --- |
| 1 | 0.00 | 100.0% | 61.6% | 0.0% |
| 1 | 0.50 | 34.3% | 86.4% | 65.7% |
| 1 | 0.70 | 13.8% | 95.7% | 86.2% |
| 1 | 0.90 | 6.5% | 99.7% | 93.5% |
| 2 | 0.00 | 100.0% | 39.2% | 0.0% |
| 2 | 0.50 | 27.0% | 60.6% | 73.0% |
| 2 | 0.70 | 15.4% | 81.2% | 84.6% |
| 2 | 0.90 | 10.8% | 93.8% | 89.2% |