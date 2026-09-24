# What other people got on this dataset

Three public Kaggle notebooks use the same ticket dataset. They are worth citing
in the final report, and two of them are worth citing as cautionary examples,
because the headline numbers do not mean what they appear to mean.

## The one that matches our task

**"Support Queue Assignment: BERT-Based Classification"**, jeevabharathis.
Same dataset, same target, queue predicted from subject and body, using
bert-base-uncased with a classification head.

| Stage | Training accuracy | Test accuracy |
| --- | --- | --- |
| After 3 epochs | 50.1% | 44.3% |
| After 8 epochs | 88.6% | 53.2% |

Their final test accuracy is 53.2%, against our 68.3%. This is the only public
result we found that is measuring the same thing we are, and we are 15 points
ahead of it.

It is also a textbook picture of overfitting. Training accuracy climbed to 88.6%
while test accuracy moved from 44.3% to 53.2%, and the test loss went up from
1.56 to 2.20 over the same stretch. The author says plainly that the result did
not meet expectations. Worth citing in our report as evidence that this task is
genuinely hard rather than evidence that we did something wrong.

## The one that reports 98.8%

**"Auto-tagging-support-tickets-llm"**, RAMEEN. Same dataset, different task.

It predicts tags rather than queues. It keeps the 20 most common tags from the
tag_1 through tag_3 columns and asks for the top 3 tags per ticket. Tickets
carry 2.59 valid tags on average. The headline metric is hit rate, meaning at
least one of the three predicted tags appears anywhere in the true tag list.

We ran that metric on our copy of the data to see what it is worth.

| Model | Hit rate | Overlap ratio |
| --- | --- | --- |
| Always guess the three most common tags | 60.2% | 31.3% |
| Three random tags | 35.5% | |
| Their fine-tuned DistilBERT | 98.8% | 84.5% |

A model with no intelligence at all, one that ignores the ticket and answers
Technical, Performance and Bug every single time, scores 60.2% on their headline
metric. So 98.8% sits on a floor of 60, not on a floor of zero. Their stricter
overlap figure of 84.5% is the honest one and it is a good result, but it is
three guesses against 2.59 right answers out of 20 candidates.

For comparison, if we score our own queue model the same lenient way, asking
only whether the correct queue appears in our top 3, we get 88.7% rather than
68.3%. Same model, same predictions, different scoring rule.

They also did not remove duplicate tickets, so their random split carries the
same leak that inflated our own midterm number.

## The one that reports 92.5%

**"Multilingual Customer Support Tickets using XLM-R"**, Muhammad Faizan. The
most upvoted notebook on the dataset, with a bronze medal.

It reports 92.5% accuracy on queue prediction, which looks like it beats
everything. The classification report at the bottom gives the reason to be
careful. The support column totals 40. The test set is forty tickets, and the
file it loads is a small sample version of the dataset rather than the 20,000
row release we use.

At that size the margin of error is roughly plus or minus 8 points, so the
result is not comparable to a number measured on thousands of tickets.

## Why this matters for our report

Our 68.3% is not a weak number. Measured the same way on the same task, the only
comparable public attempt gets 53.2%.

It also shows why the evaluation section of our report is worth writing
carefully. Across three notebooks on one dataset there are three different
tasks, three different metrics, and test sets ranging from 40 tickets to
thousands. Two of the three report numbers above 90% and neither is measuring
single-label queue routing on a deduplicated split. The number only means
something when the task, the metric and the split are stated next to it.
