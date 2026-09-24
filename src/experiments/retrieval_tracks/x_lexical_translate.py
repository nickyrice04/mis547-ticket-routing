"""Track 'lexical': translate the genuinely German tickets (language == "de", ten shared queues) to English on CPU
with the fixed pretrained Helsinki-NLP/opus-mt-de-en model (CTranslate2 int8), so a sparse English retrieval pool can
use them. Input rows come only from the parquet's language == "de" rows, which are in neither train nor test.
Output: data/x_lexical/extra_de_translated.jsonl  (resumable)."""
import json, os, re, sys, time
import pyarrow.parquet as pq
import ctranslate2
from transformers import AutoTokenizer
from common import ROOT, clean, load_meta
from experiments.retrieval_tracks.x_lexical_extra import looks_english

OUT = ROOT / "data/x_lexical/extra_de_translated.jsonl"
LIMIT = int(os.environ.get("LIMIT", 0))
_SENT = re.compile(r"(?<=[.!?])\s+")


def main():
    l2i = load_meta()["label2id"]
    t = pq.read_table(ROOT / "data_tickets.parquet").to_pydict()
    seen, rows = set(), []
    for i in range(len(t["queue"])):
        if t["language"][i] != "de" or t["queue"][i] not in l2i:
            continue
        s, b = t["subject"][i] or "", t["body"][i] or ""
        key = clean(s, b)
        if not key or key in seen or looks_english(key):
            continue
        seen.add(key)
        rows.append({"id": i, "subject": s, "body": b, "label": l2i[t["queue"][i]], "type": t["type"][i], "priority": t["priority"][i]})
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in open(OUT)}
    rows = [r for r in rows if r["id"] not in done]
    if LIMIT: rows = rows[:LIMIT]
    print(f"{len(done)} done, {len(rows)} to translate", flush=True)
    tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    tr = ctranslate2.Translator(str(ROOT / "models/x_lexical/opus-mt-de-en-ct2"), device="cpu", inter_threads=1, intra_threads=3, compute_type="int8")

    def translate(sents):
        src = [tok.convert_ids_to_tokens(tok.encode(s[:1500])) for s in sents]
        res = tr.translate_batch(src, beam_size=2, max_batch_size=2048, batch_type="tokens", max_decoding_length=256, repetition_penalty=1.1)
        return [tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True) for r in res]

    t0 = time.time(); CH = 64
    with open(OUT, "a") as f:
        for s0 in range(0, len(rows), CH):
            chunk = rows[s0:s0 + CH]
            sents, owner = [], []
            for k, r in enumerate(chunk):
                if r["subject"].strip():
                    sents.append(r["subject"].strip()); owner.append((k, "s"))
                for p in _SENT.split(r["body"].replace("\\n", " ").replace("\n", " ").strip()):
                    if p.strip():
                        sents.append(p.strip()); owner.append((k, "b"))
            out = translate(sents)
            subj = [""] * len(chunk); body = [[] for _ in chunk]
            for (k, kind), o in zip(owner, out):
                if kind == "s": subj[k] = o
                else: body[k].append(o)
            for k, r in enumerate(chunk):
                f.write(json.dumps({"id": r["id"], "text": clean(subj[k], " ".join(body[k])), "label": r["label"],
                                    "type": r["type"], "priority": r["priority"]}) + "\n")
            f.flush()
            if (s0 // CH) % 10 == 0:
                el = time.time() - t0
                print(f"{s0 + len(chunk)}/{len(rows)} {el:.0f}s  eta {el / (s0 + len(chunk)) * (len(rows) - s0 - len(chunk)) / 60:.1f} min", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
