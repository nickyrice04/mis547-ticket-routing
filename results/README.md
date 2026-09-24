# Results

One JSON per system on the same 4,750 test tickets, with accuracy, macro-F1,
the confidence threshold table and, for the final routers, accuracy by
similarity to the nearest training ticket and per-queue recall. `*_probs.npy`
are the test probabilities behind each JSON.

| File | System | Accuracy |
| --- | --- | --- |
| `1_tfidf_logreg.json` | TF-IDF + logistic regression, the midterm baseline | 61.6% |
| `1b_mlp_256.json`, `1c_mlp_512_256.json`, `1d_complement_nb.json` | small networks and Naive Bayes on the same features | 68.3%, 68.4%, 46.8% |
| `2_distilbert.json`, `2b_*`, `2c_distilbert_12ep.json` | DistilBERT, three training recipes, the last one converged | 39.2%, 56.3%, 67.7% |
| `3_roberta_base.json`, `7_lstm.json` | RoBERTa and a BiLSTM | 52.3%, 52.4% |
| `4_qwen_0_5b.json`, `6_gemma3_4b_*.json` | LLMs fine-tuned as classifiers | 42.8%, up to 43.2% |
| `5_*_zeroshot.json`, `8_deberta_*_zeroshot.json` | zero-shot, no training on our labels | 19 to 31% |
| `12_stack_semantic.json` | retrieval + embeddings + stacker, English only | **83.1%** |
| `12b_stack_german.json` | the same with the translated German pool | **91.8%** |

Other files, by story chapter.

- `sweep.json`, `learning_curve.json`, `bench_*.json`, `SUMMARY.md`, the
  width sweep, the data learning curve, memory and latency, and the report
  builder's output.
- `synthetic_comparison.json`, `qwen_think_comparison.json`,
  `synthetic_eval.json`, `sibling_eval.json`, `generator_realism.json`, every
  synthetic data comparison, all on the validation protocol, with paired
  bootstrap intervals.
- `german_curve.json`, the first German attempt through embeddings.
- `x_*` files, the five research tracks and their independent verifications.
- `*.log`, the training logs, kept because the loss curves are part of the
  evidence.
