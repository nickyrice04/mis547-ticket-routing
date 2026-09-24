"""Pull every tier's results into one table.

Run after the training scripts and bench.py. Writes results/SUMMARY.md, which is
the table that goes into the final report.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load(pattern: str) -> dict:
    """Every result JSON matching the pattern, keyed by its tier name."""
    out = {}
    for path in sorted(RESULTS.glob(pattern)):
        data = json.loads(path.read_text())
        out[data["tier"]] = data
    return out


def main() -> None:
    train = load("[0-9]*.json")
    bench = {k.replace("bench_", ""): v for k, v in load("bench_*.json").items()}

    lines = ["# Model scaling study", "", "## Accuracy against size", "",
             "| Tier | Model | Params (M) | Accuracy | Macro-F1 | Train time | Device |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for tier, d in sorted(train.items()):
        mins = d["train_seconds"] / 60
        lines.append(
            f"| {tier.split('_')[0]} | {d['model']} | {d.get('params_millions','-')} | "
            f"{d['accuracy']*100:.1f}% | {d['macro_f1']:.3f} | "
            f"{mins:.1f} min | {d.get('train_device','cpu')} |"
        )

    if bench:
        lines += ["", "## Cost to serve one ticket", "",
                  "| Tier | Disk (MB) | RAM held (MB) | p50 at 1 thread | p50 at 2 threads | p95 at 2 threads |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for tier, d in sorted(bench.items()):
            lat = d["latency"]
            one = lat.get("1_thread", {})
            two = lat.get("2_thread", one)
            lines.append(
                f"| {tier.split('_')[0]} | {d['disk_mb']:.0f} | {d['rss_after_load_mb']:.0f} | "
                f"{one.get('p50_ms','-')} ms | {two.get('p50_ms','-')} ms | {two.get('p95_ms','-')} ms |"
            )

    lines += ["", "## Confidence threshold", "",
              "How much of the queue each model can route on its own, and how often it is right.", "",
              "| Tier | Threshold | Auto-routed | Accuracy when auto-routed | Sent to a person |",
              "| --- | --- | --- | --- | --- |"]
    for tier, d in sorted(train.items()):
        for row in d.get("thresholds", []):
            if row["threshold"] in (0.0, 0.5, 0.7, 0.9):
                acc = row["accuracy_on_routed"]
                lines.append(
                    f"| {tier.split('_')[0]} | {row['threshold']:.2f} | {row['coverage']*100:.1f}% | "
                    f"{acc*100:.1f}% | {row['sent_to_human']*100:.1f}% |" if acc else ""
                )

    out = RESULTS / "SUMMARY.md"
    out.write_text("\n".join(l for l in lines if l is not None))
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
