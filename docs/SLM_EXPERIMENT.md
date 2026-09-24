# Can a language model route tickets without being trained on them?

Run on September 20th, on an Apple M5 Max using MLX, which is Apple's array
framework for Apple silicon. Models came from the `mlx-community` organization
on HuggingFace in 4-bit form, which is the same quantized format Google ships
through its AI Edge Gallery for phones.

## What was tested

No fine-tuning anywhere in this file. The model is handed the ten queue names
and asked which queue each ticket belongs to.

| Model | Parameters | Prompt | Accuracy | Macro-F1 | Per ticket |
| --- | --- | --- | --- | --- | --- |
| Gemma 3 4B, 4-bit | 4B | zero-shot | 20.6% | 0.174 | 68 ms |
| Gemma 3 4B, 4-bit | 4B | one example per queue | 18.8% | 0.136 | 75 ms |
| Qwen3 0.6B, 4-bit | 0.6B | zero-shot | unusable | unusable | 24 ms |

Against the models that were trained on this data:

| Model | Accuracy | Macro-F1 | Per ticket | Training |
| --- | --- | --- | --- | --- |
| TF-IDF + neural net | 68.3% | 0.680 | 0.2 ms | 1.7 min |
| DistilBERT fine-tuned | 67.7% | 0.658 | 17.2 ms | 31.6 min |
| TF-IDF + logistic regression | 61.6% | 0.625 | 1.2 ms | 7 sec |

## Fine-tuned, which is the fair comparison

Everything above is zero-shot, which compares a trained small model against an
untrained large one. That is not a fair fight, so the same Gemma was fine-tuned
on our tickets with LoRA and measured on the same 500.

| Gemma 3 4B, 4-bit | Accuracy | Macro-F1 | Per ticket |
| --- | --- | --- | --- |
| zero-shot | 20.6% | 0.174 | 68 ms |
| one example per queue | 18.8% | 0.136 | 75 ms |
| LoRA fine-tuned, 1,200 iterations | 36.2% | 0.188 | 184 ms |

Training details: 8 layers adapted, batch of 4, learning rate 1e-4, 1,200
iterations over 6,000 tickets, about 8 minutes, peak memory 5.9 GB. Validation
loss ended at 0.749.

Fine-tuning nearly doubled zero-shot accuracy, which confirms that the failure
was about not knowing our filing conventions rather than about language. It also
did not come close to the 68.3% of a neural network that trains in 1.7 minutes.

Be careful with that last comparison. At 1,200 iterations and a batch of 4, this
model has seen 4,800 examples, which is a quarter of one pass through the
training set. DistilBERT needed twelve full passes before it stopped improving.
This number should be read as undertrained rather than as a ceiling, and the
report should say so. A run long enough to match DistilBERT's budget would take
roughly six hours on this laptop.

The macro-F1 of 0.188 against 36.2% accuracy says it is leaning on the large
queues and mostly ignoring the small ones, which is the same behavior every
undertrained model in this study showed.

## The finding

A 4B parameter language model that has never seen our labels scores 20.6%. A
neural network that trains in under two minutes scores 68.3%. The language model
is roughly 260 times larger and gets a third of the accuracy. Teaching it our
labels for eight minutes brings it to 36.2%, still about half.

Adding one example per queue to the prompt made it slightly worse, not better,
so this is not a prompt that needs a little more tuning.

## A bug worth recording, because it nearly became a result

The first fine-tuned evaluation returned 3.6% accuracy with 481 of 500 replies
unparseable. The model was not broken. MLX wraps prompt and completion pairs in
the model's chat template during LoRA training, so a fine-tuned model that is
handed a bare prompt at inference sees a format it never trained on and emits
end-of-sequence immediately, returning an empty string.

Applying the chat template at inference took the same adapter from 3.6% to
36.2%. Nothing about the model changed. This is the same class of mistake as
training and serving with different preprocessing, which is the training and
serving skew the midterm report already warns about, and it is a good concrete
example to cite there.

## Why it fails, which matters more than that it fails

The model is not failing at English. It is failing at our filing conventions.

Three of our queues are Technical Support, Product Support and IT Support.
Those are not universal categories. This dataset decides that a request to
upgrade an integration goes to IT Support rather than Technical Support, and
nothing in the ticket text says so. It is a convention of whoever labeled the
data. A model trained on the open internet has no way to know it, and no amount
of parameters will tell it.

The trained models learn that convention from 19,000 examples. That is the whole
gap. It is a task adaptation problem, not a model capability problem.

## Qwen3 0.6B could not follow the instructions at all

Worth recording separately, because it is a capability floor rather than an
accuracy result. Asked to reply with a team number per ticket, the 0.6B model
returned the placeholder text from the instructions and then listed the queue
names in order, ignoring the tickets. Three different output formats were tried.
Gemma 3 4B followed all three correctly on the first attempt.

The lesson for an architecture document is that instruction following has a size
floor. Below roughly a billion parameters, you cannot rely on a model to return
a structured answer, which means you cannot put it in a pipeline.

## Prompt efficiency, and a control that matters

Sending ten tickets per prompt rather than one makes the instructions and the
queue list get read once for ten tickets instead of once each. That took the
cost from 215 ms per ticket to 68 ms, a factor of three.

The obvious worry is that batching confuses the model and costs accuracy. It
does not. One ticket per prompt scored 19.0% and ten per prompt scored 20.6%,
which is inside the noise of a 100 to 500 ticket sample. The efficiency is free.

## What this means for hosting

This is where the result becomes a cloud computing argument rather than a
machine learning one. An earlier draft of this document claimed the 4B model
needed a GPU droplet at roughly $550 a month. That claim was an estimate and it
was wrong, so we measured it instead.

### Measured on our own droplet

The same model in 4-bit form was run on `mis547-project-team3`, which is a 2
vCPU, 4 GB droplet costing $24 a month, using llama.cpp on the CPU.

| | |
| --- | --- |
| Model file | 2.49 GB, Q4_K_M quantization |
| Peak memory while answering | 3.35 GB |
| Fits in 4 GB | yes, without swapping |
| Load time | 6.2 seconds |
| Median time per ticket | 10.4 seconds |

Memory was never the wall. Quantization takes a 4B model from roughly 8 GB in
16-bit down to 2.5 GB, and it fits a droplet we already pay for.

Latency is the wall. Against 0.2 ms for the small neural network on the same
class of hardware, the language model is about fifty thousand times slower.

| Option | Accuracy | Time per ticket | Droplet | Monthly |
| --- | --- | --- | --- | --- |
| TF-IDF + neural net | 68.3% | 0.2 ms | 2 vCPU, 4 GB | $24 |
| DistilBERT | 67.7% | 17 ms | 2 vCPU, 4 GB | $24 |
| Gemma 3 4B, 4-bit, on CPU | 20.6% | 10,400 ms | 2 vCPU, 4 GB | $24 |

At 3,000 tickets a day, 10.4 seconds each is about 8.7 hours of continuous
compute on both vCPUs. That is workable as an overnight batch and impossible
while a customer waits for the page to respond. A GPU droplet at $0.76 an hour
would fix the speed, and that is where the $550 a month figure genuinely
applies, but it buys speed rather than accuracy.

## The fair caveat

Accuracy is not the only axis. A zero-shot model needs no labeled data and no
retraining. If the bank opens an eleventh queue tomorrow, the trained models
need new labels and a new training run, while the language model needs one more
line in its prompt. For a desk with no labeled history at all, 21% from nothing
may beat 0% from a model that cannot be trained yet.

That is a real advantage and the report should say so. It is just not worth
$526 a month once you have 19,000 labeled tickets sitting in a database.
