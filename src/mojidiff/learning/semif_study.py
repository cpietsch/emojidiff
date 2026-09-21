"""Run the SemIf decision readout over the owned editing-request set and write a report.

One frozen instruction-tuned model, one forward pass per decision, the softmax over
the answer letters. Every decision is scored twice - once with the options in their
declared order and once reversed - because a readout that follows the letter rather
than the option is a readout of nothing, and the two must agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from mojidiff.learning.semif import OPERATIONS, decision_set, reordered, score, summarise
from mojidiff.representation.codec_study import _write_bytes_artifact


def run_semif_study(
    model_source: str, revision: str, report_root: Path, *, criteria: dict[str, float]
) -> dict[str, Any]:
    import torch
    import transformers

    started = time.perf_counter()
    config = transformers.AutoConfig.from_pretrained(model_source, revision=revision)
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_source, revision=revision)
    cls: Any = transformers.AutoModelForCausalLM
    if getattr(text_config, "model_type", "") in {"qwen3_5", "qwen3_5_text"}:
        cls = transformers.Qwen3_5ForCausalLM
    model: Any = cls.from_pretrained(
        model_source,
        revision=revision,
        config=text_config,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
    )
    model.eval()
    load_seconds = time.perf_counter() - started

    results: list[dict[str, Any]] = []
    agreements = 0
    forward_seconds = 0.0
    for decision in decision_set():
        clock = time.perf_counter()
        forward = score(model, tokenizer, decision.state, OPERATIONS)
        backward = score(model, tokenizer, decision.state, reordered(OPERATIONS))
        forward_seconds += time.perf_counter() - clock
        agreements += int(forward["choice"] == backward["choice"])
        results.append(
            {
                "id": decision.id,
                "kind": decision.kind,
                "expected": decision.expected,
                "choice": forward["choice"],
                "choice_reversed": backward["choice"],
                "probabilities": dict(
                    zip(forward["option_ids"], forward["probabilities"], strict=True)
                ),
                "input_tokens": forward["input_tokens"],
                "state": decision.state,
            }
        )
    summary_stats = summarise(results)
    order_invariance = agreements / len(results)
    checks = {
        "accuracy": (summary_stats["accuracy"] or 0.0) >= criteria["min_accuracy"],
        "clarify_recall": (summary_stats["clarify_recall"] or 0.0)
        >= criteria["min_clarify_recall"],
        "order_invariance": order_invariance >= criteria["min_order_invariance"],
    }
    rows_payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in results
    )
    summary = {
        "schema_version": 1,
        "study": "semif-decision-readout",
        "model": {
            "source": model_source,
            "revision": revision,
            "dtype": "bfloat16",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        },
        "readout": (
            "native final-position logits restricted to the answer letters; softmax over letters"
        ),
        "operations": [identifier for identifier, _ in OPERATIONS],
        **summary_stats,
        "order_invariance": order_invariance,
        "timing": {
            "load_seconds": load_seconds,
            "mean_decision_seconds": forward_seconds / (2 * len(results)),
        },
        "criteria": criteria,
        "checks": checks,
        "predeclared_outcome": "passed" if all(checks.values()) else "falsified",
        "decisions_sha256": hashlib.sha256(rows_payload).hexdigest(),
    }
    _write_bytes_artifact(report_root / "decisions.jsonl", rows_payload)
    _write_bytes_artifact(
        report_root / "summary.json",
        (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.8)
    parser.add_argument("--min-clarify-recall", type=float, default=0.7)
    parser.add_argument("--min-order-invariance", type=float, default=0.9)
    args = parser.parse_args()
    summary = run_semif_study(
        args.model,
        args.revision,
        args.report_root,
        criteria={
            "min_accuracy": args.min_accuracy,
            "min_clarify_recall": args.min_clarify_recall,
            "min_order_invariance": args.min_order_invariance,
        },
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "accuracy",
                    "per_operation",
                    "per_kind",
                    "order_invariance",
                    "clarify_recall",
                    "clarify_false_alarms",
                    "checks",
                    "predeclared_outcome",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
