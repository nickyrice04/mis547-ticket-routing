# The retrieval research tracks

Once it was clear that routing on this dataset is about finding a ticket's relatives, five
research tracks ran in parallel on the same protocol (train on the core of the training
set, evaluate on the validation slice, never touch the test set). Every file here belongs
to one track, named by its prefix. Two tracks produced the final routers, which now live
in `src/final/`. The rest is kept because the negative results and the diagnostics are
part of the evidence.

These scripts are research code. They cache large intermediate arrays under `data/x_*/`
(not in git) and many assume an earlier script in the same track has run. Three of them
run their experiment on import rather than under a `__main__` guard:
`x_lexical_cv_final.py`, `x_lexical_eval_val.py`, `x_semantic_views.py`.

| Prefix | Question | Outcome |
| --- | --- | --- |
| `x_semantic_*` | Do sentence embeddings find relatives that word overlap misses, and can a learned stacker combine them? | Yes to the stacker (78.4% validation, 83.1% test), which became `final/router_english_only.py`. Dense 1-NN alone was no better than TF-IDF, "hidden siblings" are about 1% of tickets, and a contrastive fine-tune added nothing. |
| `x_german_*` | Do the German tickets, translated, add relatives? | Yes, 89.9% validation, 91.8% test, which became `final/router.py`. The remaining files here are the pool-size curve, extra channels, and the guard sensitivity sweep. |
| `x_lexical_*` | How far does sparse retrieval go with BM25, character n-grams, k-NN voting and stacking? | About +1 to +1.6 over plain 1-NN from voting rules. The learned stack was the gain, which the semantic track covered. Stopped at the session limit. |
| `x_classifier_*` | Can one parametric model on sparse features match 1-NN, so serving needs no index? | Kernel and conjunction-feature models got close inside core CV but the track stopped at the session limit before a validation confirmation. |
| `x_unfamiliar_*` | Is there any learnable signal for tickets with no relative in the pool? | Very little. A model given the true metadata reaches 40% on them, text models 38 to 43%. The ceiling analysis in `x_unfamiliar_ceiling.py` is the evidence behind "accuracy is coverage". |
| `x_verify_*` | Independent re-runs by a skeptic agent | Both final numbers reproduced to the bit, shuffled labels collapsed to 29%, no leakage found. |

## Files that matter most

- `x_semantic_common.py`, `x_semantic_stack.py`, `x_semantic_cv2.py`, `x_semantic_cv3.py`: the stacker's design and its model selection inside core.
- `x_semantic_loo_check.py`: shows leave-one-out inside core mimics validation-vs-core, which is why the stacker's training rows are honest.
- `x_semantic_poolcurve.py`: accuracy against pool size, the basis for the "more real data" argument.
- `x_german_eval.py`: guard sensitivity (how much of the German gain survives a strict leak filter).
- `x_unfamiliar_ceiling.py`, `x_unfamiliar_oracle.py`: the floor for tickets without a relative.
- `x_verify_german_twins*.py`: the search for translation twins of evaluation tickets.
