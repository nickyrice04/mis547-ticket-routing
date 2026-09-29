# What other people got on this dataset

Three public Kaggle notebooks work with the same ticket data, the Kaggle release of
Bueck's dataset ([tobiasbueck/multilingual-customer-support-tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)).
We read their code and re-scored their headline metrics on our copy of the data, because a
number only means something when the task, the metric and the test set are stated next to it.
Full citations are in [REFERENCES.md](REFERENCES.md).

| Notebook | Task | Their headline number | What it compares to |
| --- | --- | --- | --- |
| [Jeevabharathi S (2025)](https://www.kaggle.com/code/jeevabharathis/support-queue-assignment-bert-based-classification), BERT | one queue per ticket, the same task as ours | 53.2% test accuracy | our final router, 91.8% (83.1% using English tickets only) |
| [Rameen (2026)](https://www.kaggle.com/code/ramfat/auto-tagging-support-tickets-llm), DistilBERT | three tags out of twenty, a hit if any one is right | 98.8% hit rate | a model that ignores the ticket scores 60.2% on the same metric |
| [Muhammad Faizan (2024)](https://www.kaggle.com/code/muhammadfaizan65/multilingual-customer-support-tickets-using-xlm-r), XLM-R | one queue per ticket | 92.5% accuracy | measured on 40 test tickets, plus or minus 8 points |

## The one that matches our task

**"Support Queue Assignment: BERT-Based Classification"**, Jeevabharathi S, April 16, 2025.
Same dataset, same target, queue predicted from subject and body, using bert-base-uncased
with a classification head.

| Stage | Training accuracy | Test accuracy |
| --- | --- | --- |
| After 3 epochs | 50.1% | 44.3% |
| After 8 epochs | 88.6% | 53.2% |

Their final test accuracy is 53.2%. On our deduplicated test set of 4,750 tickets the final
router gets 91.8%, the English-only version 83.1%, and our simple bag-of-words network
(the midterm baseline) 68.3%. This is the only public result we found that measures the
same thing we do.

It is also a textbook picture of overfitting. Training accuracy climbed to 88.6% while test
accuracy moved from 44.3% to 53.2%, and the test loss went up from 1.56 to 2.20 over the
same stretch. The author says plainly that the result did not meet expectations. It matches
what we saw ourselves. Fine-tuned transformers did not beat the bag-of-words network here
(DistilBERT 67.7%, RoBERTa 52.3%), because the label lives in the family a ticket belongs
to rather than in its wording ([MODEL_STORY.md](MODEL_STORY.md)). Retrieval, which looks up
the family directly, is what moved the number.

## The one that reports 98.8%

**"Auto-tagging-support-tickets-llm"**, Rameen, 2026. Same dataset, different task.

It predicts tags rather than queues. It keeps the 20 most common tags from the tag_1
through tag_3 columns and asks for the top 3 tags per ticket. Tickets carry 2.59 valid tags
on average. The headline metric is hit rate, meaning at least one of the three predicted
tags appears anywhere in the true tag list.

We ran that metric on our copy of the data to see what it is worth.

| Model | Hit rate | Overlap ratio |
| --- | --- | --- |
| Always guess the three most common tags | 60.2% | 31.3% |
| Three random tags | 35.5% | |
| Their fine-tuned DistilBERT | 98.8% | 84.5% |

A model with no intelligence at all, one that ignores the ticket and answers Technical,
Performance and Bug every single time, scores 60.2% on their headline metric. So 98.8% sits
on a floor of 60, not on a floor of zero. Their stricter overlap figure of 84.5% is the
honest one and it is a good result, but it is three guesses against 2.59 right answers out
of 20 candidates.

Scoring rules move numbers a lot. If we score our own routers the same lenient way, asking
only whether the correct queue is among our top 3, the final router gets 96.8% instead of
91.8%, and the English-only router 93.0% instead of 83.1%. Same predictions, different rule.
The report uses the strict single-answer accuracy throughout.

They also did not remove duplicate tickets, so their random split carries the same leak
that inflated our own midterm number.

## The one that reports 92.5%

**"Multilingual Customer Support Tickets using XLM-R"**, Muhammad Faizan, July 17, 2024.
The most upvoted notebook on the dataset, with a bronze medal.

It reports 92.5% accuracy on queue prediction, which looks like it beats everything. The
classification report at the bottom gives the reason to be careful. The support column
totals 40. The test set is forty tickets, and the file it loads is an early, small version
of the dataset from mid-2024 rather than the full release we use. The Kaggle version history
shows the dataset only grew to 2,000 tickets in December 2024.

At that size the margin of error is roughly plus or minus 8 points. Our final router's 91.8%
looks similar, but it is measured on 4,750 deduplicated tickets, where the margin of error is
under 1 point.

## Why this matters for our report

Measured the same way on the same task, the only comparable public attempt gets 53.2% and
our final router gets 91.8%.

It also shows why the evaluation section of the report is worth writing carefully. Across
three notebooks on one dataset there are three different tasks, three different metrics,
and test sets from 40 tickets to thousands. Two of the three report numbers above 90% and
neither is measuring single-label queue routing on a deduplicated split. A number only
means something when the task, the metric and the split are stated next to it.
