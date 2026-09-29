"""Collect the semantic track's headline numbers into results/x_semantic_summary.json (no model fitting here)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]   # retrieval_tracks -> experiments -> src -> repository root
R = ROOT / "results"
final = json.loads((R / "x_semantic_final_run.json").read_text())
cv1 = json.loads((R / "x_semantic_cv1.json").read_text())
pool = json.loads((R / "x_semantic_poolcurve.json").read_text())
summary = {
    "track": "semantic",
    "protocol": "train on core (16,147), evaluate on validation (2,850); all model selection by CV / leave-one-out inside core",
    "final_module": "src/final/router_english_only.py",
    "headline": {
        "val_accuracy": final["val_accuracy"], "val_macro_f1": final["val_macro_f1"],
        "runtime_minutes_core_to_val": final["runtime_minutes"],
        "reference_lex_1nn_val": final["lex_1nn_acc"], "reference_fused_1nn_val": final["fused_1nn_acc"],
        "reference_simple_hybrid_val_with_LR": final["simple_hybrid_acc"],
        "loo_cv_inside_core_same_stack": 0.7852,
    },
    "bucket_table_validation": final["buckets"],
    "step1_pretrained_embedders_1nn_validation": {
        "lexical_tfidf": 0.7449, "multilingual-e5-base": 0.7509, "bge-base-en-v1.5": 0.7491, "all-mpnet-base-v2": 0.7358,
        "gte-base": 0.7432, "bge-large-en-v1.5": 0.7474, "e5-large-v2": 0.7540, "gte-modernbert-base": 0.7512,
        "all-MiniLM-L12-v2": 0.7319, "fused lex + 5*mean(4 dense) 1-NN": final["fused_1nn_acc"]},
    "step1_loo_1nn_inside_core": {"lex": 0.7432, "char_tfidf": 0.7458, "e5base": 0.7513, "bge": 0.7409, "mpnet": 0.7330, "gte": 0.7396,
                                  "bgelarge": 0.7430, "e5large": 0.7508, "gtemodern": 0.7518, "minilm": 0.7397,
                                  "fused4": 0.7562, "fused8": 0.7552},
    "score_fusion_5fold_cv_inside_core": cv1,
    "stack_loo_cv_inside_core": {
        "lex kNN feats only": 0.7609, "lex kNN + tfidf LR": 0.7698, "pre4 (lex + 4 dense + fused + 2 LR) [FINAL]": 0.7852,
        "pre4 + char/subject/body lexical views": 0.7852, "pre8 (8 embedders)": 0.7873, "pre8 + lexical views": 0.7876,
        "pre4 + MLP(tfidf) probs": 0.7858, "pre4 + MLP(tfidf) + MLP(8 embedders) probs": 0.7852,
        "meta hyper-parameter variants (6)": "0.7834 - 0.7860"},
    "step2_contrastive_finetune": {
        "recipe": "bge-base-en-v1.5, MNRL-style loss (scale 20), triplets mined only from 80% of core: positive = same queue & lexical cos >= 0.4 "
                  "within top-10 fused neighbours, hard negative = other-queue ticket in top-10; same-queue in-batch candidates masked; "
                  "3 epochs, 405 steps, batch 48, lr 2e-5, max_len 192, 27.3 min on shared MPS",
        "evaluated_on": "20% dev slice of core never seen by the fine-tune (3,230 queries, pool = all other core tickets)",
        "dev_1nn": {"lex": 0.7502, "bge pretrained": 0.7446, "e5base pretrained": 0.7638, "fused4": 0.7641, "bge fine-tuned": 0.7656},
        "dev_stack": {"pre4 stage-1 stack": 0.7981, "dev-only meta pre4 feats": 0.7921, "dev-only meta pre4 + ft view": 0.7943,
                      "two-stage P1 + ft view": 0.7970, "two-stage control (P1 only)": 0.7976},
        "verdict": "fine-tuning lifts the single embedder's 1-NN by +2.1 points on unseen tickets but adds nothing once lexical + pretrained "
                   "dense views are stacked; not used in the final module. Second fine-tune slot deliberately not spent."},
    "diagnostics": {
        "hidden_siblings": "when the dense 1-NN differs from the lexical 1-NN and its lexical cosine is < 0.3 (4,828 core tickets), it shares the queue "
                           "only 30-33% of the time = base rate for same-scenario tickets; only ~185 tickets (1.1%) with a large dense margin reach 66%",
        "family_size": "P(same queue) by fused neighbour rank inside core: 0.69, 0.44, 0.32, 0.29, 0.29 ... -> families are mostly pairs/triples",
        "metadata_oracle_low_band": "LogReg on the TRUE type+priority+version+tags predicts queue at only 38-41% for tickets with nearest lexical "
                                    "cosine < 0.3 (text LR 38-40%, stack 43%); the queue of a ticket without a sibling in the pool is near the noise floor",
        "per_class_recall_low_band": "only Billing (0.72), Outages (0.45) and the majority class Technical Support (0.77) are recoverable; others 0.06-0.24",
        "pool_size_curve_inside_core": pool,
        "pool_size_note": "1-NN accuracy grows ~2.1-2.4 points per extra 1,000 pool tickets (lex: 0.632 @11.2k, 0.679 @13.1k, 0.743 @16.1k LOO). "
                          "Extrapolating that slope (with its mild decay, ~1.9 pt/1k) to the full-train pool of 18,997 suggests roughly +5 points for the "
                          "final train->test run relative to the core->validation number, i.e. about 0.82-0.84 for this stack. This is an extrapolation, "
                          "not a measurement; the test set was never touched."},
    "configs_scored_on_validation": {
        "count": 12,
        "list": ["lexical 1-NN", "8 pretrained embedders 1-NN (by-product printout of the embedding script)", "fused 1-NN (by-product)", "simple hybrid 1-NN-or-LR at 0.3 (by-product reference printed by the final module)",
                 "pre4 stack (the one configuration selected by core-internal CV; re-run once through the final module, identical 0.7835)"],
        "note": "no choice was made using validation; every selection (fusion weight, views, classifiers, meta hyper-parameters, fine-tune verdict) "
                "used CV / leave-one-out / a dev slice strictly inside core"},
}
(R / "x_semantic_summary.json").write_text(json.dumps(summary, indent=1))
print("wrote", R / "x_semantic_summary.json")
for r in final["buckets"]:
    print(r)
