"""Translate the German tickets into English so they can join the retrieval pool.

The parquet holds 16,500 German tickets in the same ten queues as the English ones. They
are rewrites of the same scenario families, so once translated they are real labelled
relatives for tickets the English data never covered (worth about nine points on test).
A word-for-word translation would not do: German word order and compound nouns would
leave the output unlike any English ticket, and the router matches on wording. So the
translation is done by Helsinki-NLP/opus-mt-de-en, an open 74M-parameter neural
translation model that is run locally and never retrained. All 16,500 tickets take about
11 minutes on the laptop GPU.

Nothing here looks at the English train, validation or test tickets. The only inputs are
the German rows of the parquet. Three steps, each resumable:

    PYTHONPATH=src python src/final/translate.py extract     # write data/x_german/german_raw.jsonl
    PYTHONPATH=src python src/final/translate.py translate   # sentence-level NMT with a resumable cache
    PYTHONPATH=src python src/final/translate.py assemble    # write data/x_german/german_translated.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from common import ROOT, clean, load_meta

OUT = ROOT / "data" / "x_german"
RAW = OUT / "german_raw.jsonl"            # the German rows, untouched
CACHE = OUT / "sentence_cache.jsonl"      # one line per translated sentence, so a rerun resumes
FINAL = OUT / "german_translated.jsonl"   # what the router reads
MODEL = "Helsinki-NLP/opus-mt-de-en"
REVISION = "1a922f3b32a8e809e17a47d4b32142d8105924e5"   # pinned, same weights everywhere
BATCH_TOKENS = 6000        # padded tokens per batch, sized for the GPU
MAX_SRC_TOKENS = 200       # longer "sentences" are split on commas before translation

# Sentence boundaries: end punctuation followed by whitespace, or a line break.
_SPLIT = re.compile(r"(?<=[.!?:;])\s+|\n+")


def best_device() -> str:
    """The fastest device available: an NVIDIA GPU (the cloud GPU droplet), Apple's GPU (the laptop), or the CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract() -> None:
    """German rows in the ten shared queues, exact duplicates (after cleaning) removed."""
    import pyarrow.parquet as pq

    labels = set(load_meta()["labels"])
    t = pq.read_table(ROOT / "data_tickets.parquet").to_pydict()
    OUT.mkdir(parents=True, exist_ok=True)
    seen, n = set(), 0
    with open(RAW, "w") as f:
        for i in range(len(t["queue"])):
            if t["language"][i] != "de" or t["queue"][i] not in labels:
                continue
            key = clean(t["subject"][i], t["body"][i])
            if not key or key in seen:
                continue
            seen.add(key)
            f.write(json.dumps({
                "row": i, "subject": t["subject"][i] or "", "body": t["body"][i] or "",
                "queue": t["queue"][i], "type": t["type"][i], "priority": t["priority"][i],
                "tags": [t[f"tag_{k}"][i] for k in range(1, 9) if t[f"tag_{k}"][i]],
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} unique German tickets in the ten shared queues to {RAW}")


def sentences(text: str) -> list[str]:
    """Split a ticket field into sentences. The parquet stores line breaks as literal backslash sequences."""
    text = (text or "").replace("\\n", "\n").replace("\\t", " ").replace("\\r", " ")
    parts = [p.strip() for p in _SPLIT.split(text)]
    return [p for p in parts if p]


def load_cache() -> dict[str, str]:
    """German sentence -> English sentence, from every earlier run."""
    cache = {}
    if CACHE.exists():
        for line in open(CACHE):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:   # a line cut short by an interrupted run
                continue
            cache[r["de"]] = r["en"]
    return cache


def translate() -> None:
    """Translate every sentence not yet in the cache, batched by token budget, longest sentences last."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    rows = [json.loads(l) for l in open(RAW)]
    cache = load_cache()
    todo = set()
    for r in rows:
        for s in sentences(r["subject"]) + sentences(r["body"]):
            if s not in cache:
                todo.add(s)
    print(f"{len(rows)} tickets, {len(cache)} sentences cached, {len(todo)} to translate", flush=True)
    if not todo:
        return

    tok = MarianTokenizer.from_pretrained(MODEL, revision=REVISION)
    device = best_device()
    model = MarianMTModel.from_pretrained(MODEL, revision=REVISION).to(device).eval()
    if device in ("mps", "cuda"):
        model = model.half()             # half precision on a GPU, about twice as fast

    # Very long "sentences" (lists, run-ons) are cut at commas so nothing is truncated.
    # A unit is (original sentence, the pieces it was cut into).
    units: list[tuple[str, list[str]]] = []
    for s in todo:
        n_tok = len(tok.encode(s))
        if n_tok <= MAX_SRC_TOKENS:
            units.append((s, [s]))
        else:
            pieces, cur = [], ""
            for part in re.split(r"(?<=,)\s+", s):
                if cur and len(tok.encode(cur + " " + part)) > MAX_SRC_TOKENS:
                    pieces.append(cur)
                    cur = part
                else:
                    cur = (cur + " " + part).strip()
            if cur:
                pieces.append(cur)
            units.append((s, pieces))

    # Pieces are sorted by length so each batch pads as little as possible. A batch grows
    # until its padded size (rows x longest row) would exceed BATCH_TOKENS.
    flat = sorted({p for _, ps in units for p in ps}, key=len)
    lengths = {p: min(len(tok.encode(p)), 256) for p in flat}
    done: dict[str, str] = {}
    t0 = time.time()
    gpu_time = 0.0
    i = 0
    out = open(CACHE, "a")
    pending = {s: ps for s, ps in units}
    while i < len(flat):
        j, longest = i, 0
        while j < len(flat):
            longest_new = max(longest, lengths[flat[j]])
            if (j - i + 1) * longest_new > BATCH_TOKENS and j > i:
                break
            longest = longest_new
            j += 1
        batch = flat[i:j]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        g0 = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, num_beams=2, max_new_tokens=int(longest * 1.6) + 12)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        gpu_time += time.time() - g0
        for p, e in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
            done[p] = e
        i = j
        # flush every full sentence whose pieces are all done, so an interrupted run loses nothing
        finished = [s for s, ps in pending.items() if all(p in done for p in ps)]
        for s in finished:
            out.write(json.dumps({"de": s, "en": " ".join(done[p] for p in pending.pop(s))},
                                 ensure_ascii=False) + "\n")
        out.flush()
        if (i // max(1, len(batch))) % 20 == 0:
            print(f"{i}/{len(flat)} pieces, {time.time()-t0:.0f}s wall, {gpu_time:.0f}s gpu", flush=True)
    out.close()
    print(f"done: {len(flat)} pieces in {time.time()-t0:.0f}s wall, {gpu_time:.0f}s gpu", flush=True)


# Tiny stop-word lists for telling an English sentence from a German one.
_EN_SW = set("the and to of is for with that this have please would could our your are we you i on it be "
             "can any my has not or by from".split())
_DE_SW = set("der die das und ist nicht ich wir sie zu mit für auf ein eine den dem von bitte können wurde "
             "dass haben es im des sich auch bei um zur zum einen einer wird sind".split())
_WORD = re.compile(r"[a-zäöüß]+")


def is_english(sentence: str) -> bool:
    """About a quarter of the rows labelled 'de' are written in English. The NMT model garbles
    English input ("inquire" -> "require"), so English sentences are passed through untouched."""
    w = _WORD.findall(sentence.lower())
    e = sum(x in _EN_SW for x in w)
    d = sum(x in _DE_SW for x in w)
    return (e >= 1 and d == 0) or (e >= 3 and e >= 3 * d)


def assemble() -> None:
    """Rebuild each ticket from its translated sentences, clean it like an English ticket, write the pool file.

    Each row keeps the parquet row number, the queue and its id, the metadata, and the
    share of sentences that were already English (a diagnostic, never a model input).
    """
    rows = [json.loads(l) for l in open(RAW)]
    cache = load_cache()
    labels = load_meta()["labels"]
    lab2id = {l: i for i, l in enumerate(labels)}
    n, missing, seen = 0, 0, set()
    to_en = lambda s: s if is_english(s) else cache[s]
    with open(FINAL, "w") as f:
        for r in rows:
            subj = sentences(r["subject"])
            body = sentences(r["body"])
            if any(s not in cache for s in subj + body):
                missing += 1
                continue
            text = clean(" ".join(to_en(s) for s in subj), " ".join(to_en(s) for s in body))
            if not text or text in seen:
                continue
            seen.add(text)
            sent = subj + body
            f.write(json.dumps({"row": r["row"], "text": text, "queue": r["queue"],
                                "label": lab2id[r["queue"]], "type": r["type"],
                                "priority": r["priority"], "tags": r["tags"],
                                "english_source_share": round(sum(map(is_english, sent)) / max(1, len(sent)), 3)},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} translated tickets to {FINAL} ({missing} skipped for missing sentences)")


class Translator:
    """Online translation for the API: one ticket at a time, same rules as the batch job.

    A German ticket that reaches the router has to look like the tickets in the pool,
    which were built by assemble(). So this applies the same steps to one ticket: split
    into sentences, pass English sentences through untouched, translate the rest with the
    same model and beam size, cut very long sentences at commas, then clean. On the
    inference droplet's CPU a typical German ticket takes one to two seconds.
    """

    def __init__(self, model_name: str = MODEL, device: str | None = None):
        from transformers import MarianMTModel, MarianTokenizer

        self.device = device or best_device()
        self.tok = MarianTokenizer.from_pretrained(model_name, revision=REVISION)
        self.model = MarianMTModel.from_pretrained(model_name, revision=REVISION).to(self.device).eval()

    def _pieces(self, s: str) -> list[str]:
        """Cut a sentence longer than MAX_SRC_TOKENS at commas, the same rule the batch job uses."""
        if len(self.tok.encode(s)) <= MAX_SRC_TOKENS:
            return [s]
        pieces, cur = [], ""
        for part in re.split(r"(?<=,)\s+", s):
            if cur and len(self.tok.encode(cur + " " + part)) > MAX_SRC_TOKENS:
                pieces.append(cur)
                cur = part
            else:
                cur = (cur + " " + part).strip()
        return pieces + ([cur] if cur else [])

    def translate_sentences(self, sents: list[str]) -> dict[str, str]:
        """German sentence -> English sentence for every sentence given."""
        import torch

        if not sents:
            return {}
        units = {s: self._pieces(s) for s in sents}
        flat = sorted({p for ps in units.values() for p in ps}, key=len)
        enc = self.tok(flat, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        longest = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            gen = self.model.generate(**enc, num_beams=2, max_new_tokens=int(longest * 1.6) + 12)
        done = dict(zip(flat, self.tok.batch_decode(gen, skip_special_tokens=True)))
        return {s: " ".join(done[p] for p in ps) for s, ps in units.items()}

    def to_english(self, subject: str, body: str) -> dict:
        """The cleaned English text the router reads, plus what happened on the way.

        Returns {"text", "language", "translated_sentences", "sentences"}. language is "de"
        when most sentences were not English, else "en".
        """
        subj, bod = sentences(subject), sentences(body)
        n = len(subj) + len(bod)
        foreign = [s for s in subj + bod if not is_english(s)]
        # English tickets in the training pool were never translated, only cleaned. So a
        # ticket that is mostly English is treated the same way, even if a short line such
        # as "VPN keeps disconnecting" has no English stop words for the detector to see.
        if len(foreign) * 2 <= n:
            return {"text": clean(subject, body), "language": "en", "translated_sentences": 0, "sentences": n}
        mapping = self.translate_sentences(foreign)
        to_en = lambda s: mapping.get(s, s)
        text = clean(" ".join(to_en(s) for s in subj), " ".join(to_en(s) for s in bod))
        return {"text": text, "language": "de", "translated_sentences": len(foreign), "sentences": n}


if __name__ == "__main__":
    {"extract": extract, "translate": translate, "assemble": assemble}[sys.argv[1]]()
