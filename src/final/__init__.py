"""The model that ships.

router.py                 fit_predict_proba(train_texts, train_labels, eval_texts), 91.8% on test
router_english_only.py    the same idea without the German pool, 83.1% on test
features.py               the five retrieval channels and the leak guard
stacker.py                the gradient-boosted model that turns neighbour features into probabilities
lib.py                    the TF-IDF vectorizer and top-k similarity helpers
embed.py                  multilingual-e5-base embeddings, cached under data/x_german/
translate.py              German to English with Helsinki-NLP opus-mt-de-en
test_once.py              the one-shot test run, refuses to run twice

These files were named x_german_* and x_semantic_final.py while the results in
results/12*_stack_*.json were produced. The code is unchanged, only the names.
"""
