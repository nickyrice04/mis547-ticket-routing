# How the model was found

Nicky Rice, September 24th. This is the story of the model, iteration by
iteration, written for the team and for the final report. Numbers are on the
deduplicated test split unless a sentence says validation.

First, a correction. The 70.5% from the midterm was wrong. I had not removed
the duplicate tickets, and about a quarter of the test set had exact copies in
training. With the duplicates removed before the split, the TF-IDF plus
logistic regression baseline is 61.6%. It gave us a starting point, but not
enough accuracy to claim an automated system.

I tried a more complex model next, a small neural network on the same bag of
words features with one hidden layer of 256 units, and saw a large increase to
68.3%. I tested wider layers of 512 and 1,024 and a two layer network of 512
and 256, and every one landed within a point of the 256 network. Naturally I
thought the next step was larger, more complex models. The goal was to justify
cloud compute through the accuracy that larger models, which need more compute
and memory to train and run, would give us.

So I tried a larger model, DistilBERT, as discussed in the midterm update. It
is not an LLM. It turns the ticket into a vector of embeddings, a numeric
representation of the words in context, and a final layer scores which area of
the business should receive the ticket. Fine tuning changed all of its weights
based on our training data. Despite the larger model I could only get 67.7%,
and it took 12 passes over the data and half an hour of GPU to get there. Back
to the drawing board. I kept following the same logic of larger models giving
better predictions. I fine tuned Qwen2.5 0.5B and Gemma 3 4B as classifiers
and both dropped to around 43%. I also tried Laya, an open decision model
built for calibrated routing. An hour of fine tuning later it gave 58% on the
validation slice and had flattened.

It became clear that larger models were not going to be good classifiers here,
and the small network was winning at 68%. That is still not high enough for a
reliable system in a business environment, and our justification for cloud use
was poor. So I started using the LLMs in other ways. I fine tuned them to
generate synthetic data, on the logic that more data would let the small
network gain accuracy. After multiple models, several kinds of generation and
a lot of prompt engineering, the best synthetic data gave the network about 3
points on the validation slice, and did nothing for the retrieval method
described below. Still not good enough.

I needed to know why this was happening and whether 70% was the ceiling. What
finally explained it was how the dataset was made. Many tickets were written
with the same meaning but different wording, and these tickets form families
that share a queue. When the model has seen a relative of a ticket, it routes
it right about 97% of the time. When it has never seen the family, it is right
about 40% of the time, even a version I handed the dataset's own tags and
priority. In other words, accuracy comes down to whether the model has seen a
relative of the ticket before. That explained why bigger models failed. If
there is nothing in the text to learn from, a bigger reader does not help. It
explained the synthetic data too. A generator can only reword what it has
seen, so it makes more of the same families and cannot label one it has never
met.

The finding also changed the model. If accuracy comes from having seen a
relative, the simplest thing is to look one up. Taking the queue of the single
most similar training ticket scored 74.5% on validation with no training at
all, ten points above the network. Building a small learned model on top of
that lookup, described at the end, reached 83.1% on the test set with English
data alone.

I also tested how much real data was worth. I set aside 15% of the training
set and trained the network on the remaining 85%, which scored 65.3%. Adding
the 2,850 real tickets back gave 3.1 points. Adding 7,700 synthetic tickets
gave 0.7. Real data was going to be huge for this dataset. The question was
where to get more of it.

Then I realized something. The dataset had 16,500 German tickets that
preprocessing had been throwing away. I had considered translating them and
worried about integrity, cultural differences, direct translations not keeping
the meaning. What I had not considered was letting a model handle the nuances
of language. A word for word translator would not have worked. German puts
verbs at the end, glues nouns into single words, and phrases things with no
direct English match, so the output reads like broken English, and since the
method depends on a translated ticket lining up with its English relatives,
broken wording would leave it unmatched. A neural translation model, a
transformer trained on millions of real sentence pairs, writes the natural
English a person would, so the tickets match. This gave us more real data. It
also justifies cloud compute on its own. The translator is an open 74 million
parameter model that we only run, never retrain, and it went through all
16,500 tickets in about 11 minutes on a GPU. A short burst of compute, on
private data that cannot go to an outside service.

Once the German tickets are translated, every labelled ticket we have,
English and translated, sits in one pool. When a new ticket comes in, the
model finds the 20 most similar tickets in that pool, once by matching words
and once by matching meaning with a sentence embedding. For each of the ten
queues it notes two things, how close the nearest ticket from that queue was
and how many of the 20 came from it. A small gradient boosted model turns
those numbers into a probability for each queue. The highest is the prediction
and its size is the confidence. The heavy lifting is looking up similar past
tickets, and the learned part on top is small, which is why it trains in
minutes and predicts in milliseconds. With the translated tickets in the pool
the final model scores 91.8% on the test set, with a macro F1 of 0.917 and
every queue between 87% and 96% recall. Now there is real justification for
an automated system.

Most of the remaining misses are the dataset itself, either scenarios with no
relative anywhere or the same scenario labelled two different ways in the
source. In a real deployment both shrink over time, because every ticket a
person corrects goes back into the pool as a real relative with a real label.
That feedback loop is the same mechanism that made the German data work, and
it is what the cloud side is built around.
