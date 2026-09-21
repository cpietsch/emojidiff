"""Gate M, experiment three: SemIf as a bounded decision layer for editing requests.

SemIf is a readout, not a model. A frozen instruction-tuned language model is shown a
state, a criterion and a lettered list of options, and one forward pass yields the
logits of the answer letters at the final position; a softmax over those letters alone
is the decision. Nothing is decoded. This module reimplements that direct readout from
the reference implementation - the same system prompt, the same JSON payload, the same
single-token slot verification - so the measurement here is of the method rather than
of a port of its code.

The decision it makes for this project is which editing operation a request asks for:
generate a new icon, recolour a selected shape, restyle its stroke, simplify the icon,
or ask for the missing detail. The evaluation set is owned and labelled by hand, with
paraphrases, negations, reordered options and requests that lack an actionable target,
because a router that cannot say "ask" is a router that guesses.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

LETTERS = "ABCDEFGHIJKLMNOP"

DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

OPERATIONS: tuple[tuple[str, str], ...] = (
    ("generate", "Create a new icon from the description; nothing existing is edited."),
    ("recolor", "Change the fill or stroke colour of the selected shape to a stated colour."),
    (
        "restyle",
        "Change the stroke of the selected shape: its width, its cap or join, or remove it.",
    ),
    ("simplify", "Reduce the icon's detail or path complexity without changing its subject."),
    (
        "clarify",
        "The request cannot be carried out as stated: the target, the value or the intent is "
        "missing or contradictory, so ask the user for the missing detail.",
    ),
)

QUESTION = (
    "Which editing operation does the user's request ask for? Choose 'ask' only when the "
    "request cannot be carried out without more information."
)


@dataclass(frozen=True)
class Decision:
    id: str
    state: dict[str, Any]
    expected: str
    kind: str
    """One of: base, paraphrase, negation, reordered, missing_target, distractor."""


def direct_messages(
    state: dict[str, Any], options: tuple[tuple[str, str], ...]
) -> list[dict[str, str]]:
    payload = {
        "evidence": state,
        "criterion": QUESTION,
        "options": [
            {"letter": LETTERS[index], "description": description}
            for index, (_, description) in enumerate(options)
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def slot_ids(tokenizer: Any, count: int) -> list[int]:
    """The answer letters must each be exactly one token that decodes back to itself."""

    result = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"answer slot {letter!r} is not one exact round-trip token")
        result.append(encoded[0])
    if len(result) != len(set(result)):
        raise ValueError("answer-slot tokens collide")
    return result


def encode_prompt(
    tokenizer: Any,
    state: dict[str, Any],
    options: tuple[tuple[str, str], ...],
    max_tokens: int = 4096,
) -> tuple[list[int], list[int], str]:
    prompt = tokenizer.apply_chat_template(
        direct_messages(state, options),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"{len(ids)} input tokens exceed {max_tokens}; no truncation allowed")
    slots = slot_ids(tokenizer, len(options))
    for letter, token in zip(LETTERS, slots, strict=False):
        if tokenizer.encode(prompt + letter, add_special_tokens=False) != ids + [token]:
            raise ValueError(f"answer boundary changes tokenization for slot {letter}")
    return ids, slots, hashlib.sha256(prompt.encode()).hexdigest()


def score(
    model: Any,
    tokenizer: Any,
    state: dict[str, Any],
    options: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    """One forward pass; the softmax over the answer letters' final-position logits."""

    import torch

    ids, slots, prompt_hash = encode_prompt(tokenizer, state, options)
    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
    }
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    selected = logits[slots].cpu().tolist()
    probabilities = softmax(selected)
    return {
        "option_ids": [identifier for identifier, _ in options],
        "probabilities": probabilities,
        "option_logits": selected,
        "choice": options[max(range(len(options)), key=lambda index: probabilities[index])][0],
        "input_tokens": len(ids),
        "prompt_sha256": prompt_hash,
    }


def reordered(options: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    """The same options with the letters assigned in reverse; the answer must not move."""

    return tuple(reversed(options))


# --------------------------------------------------------------------------- the set


def _state(request: str, selection: str | None = None, icon: str | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {"request": request}
    if icon is not None:
        state["current_icon"] = icon
    state["selected_shape"] = selection if selection is not None else "none"
    return state


def decision_set() -> tuple[Decision, ...]:
    """Fifty-two owned decisions across five operations, with the failure modes named.

    Labels are the author's and are the contract: a disagreement with them is a
    disagreement to be argued in the record, not a reason to relabel after the fact.
    """

    rows: list[Decision] = []

    def add(
        identifier: str,
        expected: str,
        kind: str,
        request: str,
        selection: str | None = None,
        icon: str | None = None,
    ) -> None:
        rows.append(Decision(identifier, _state(request, selection, icon), expected, kind))

    # generate - a new icon from a description, nothing selected.
    add("gen-1", "generate", "base", "A sleepy robot holding an apple")
    add("gen-2", "generate", "base", "Draw a hedgehog curled into a ball")
    add("gen-3", "generate", "base", "I need an icon of a lighthouse at night")
    add("gen-4", "generate", "base", "make me a cactus in a terracotta pot")
    add("gen-5", "generate", "paraphrase", "Could you come up with a picture of a paper plane?")
    add("gen-6", "generate", "paraphrase", "New icon: a steaming cup of tea with a lemon slice")
    add("gen-7", "generate", "distractor", "Draw a blue whale", None, "sun.svg")
    add(
        "gen-8",
        "generate",
        "distractor",
        "Start over with a completely different subject: a snail",
        "body",
        "hedgehog.svg",
    )
    add(
        "gen-9",
        "generate",
        "negation",
        "Don't edit the current one, I want a fresh icon of a key",
        "outline",
        "lock.svg",
    )
    add("gen-10", "generate", "base", "an origami crane, flat style")

    # recolor - a stated colour on a selected shape.
    add("rec-1", "recolor", "base", "Make the selected shape blue", "body", "fish.svg")
    add("rec-2", "recolor", "base", "Change the fill to #ff0000", "hat", "wizard.svg")
    add(
        "rec-3",
        "recolor",
        "paraphrase",
        "Can the highlighted part be green instead?",
        "leaf",
        "apple.svg",
    )
    add(
        "rec-4",
        "recolor",
        "paraphrase",
        "Swap this one to a darker skin tone",
        "face",
        "person.svg",
    )
    add("rec-5", "recolor", "base", "Turn the outline of this shape orange", "eye", "owl.svg")
    add("rec-6", "recolor", "distractor", "Colour it yellow, like the sun", "petal", "flower.svg")
    add("rec-7", "recolor", "negation", "Not red anymore - make it purple", "cape", "hero.svg")
    add("rec-8", "recolor", "base", "paint the selected stroke white", "whisker", "cat.svg")
    add("rec-9", "recolor", "paraphrase", "I'd like this bit in teal please", "wave", "ocean.svg")
    add(
        "rec-10",
        "recolor",
        "base",
        "set fill colour of selection to #92d3f5",
        "sky",
        "landscape.svg",
    )

    # restyle - the stroke's width, cap, join or presence.
    add("res-1", "restyle", "base", "Make the outline thicker", "body", "fish.svg")
    add("res-2", "restyle", "base", "Thinner stroke on this one", "frame", "picture.svg")
    add(
        "res-3",
        "restyle",
        "paraphrase",
        "Use rounded line ends for the selected path",
        "smile",
        "face.svg",
    )
    add(
        "res-4", "restyle", "paraphrase", "Remove the border from this shape", "circle", "badge.svg"
    )
    add("res-5", "restyle", "base", "Set the stroke width to 3", "arm", "robot.svg")
    add(
        "res-6",
        "restyle",
        "distractor",
        "The line looks heavy, lighten its weight",
        "antenna",
        "robot.svg",
    )
    add(
        "res-7",
        "restyle",
        "negation",
        "Don't change the colour, just make the line bolder",
        "stem",
        "flower.svg",
    )
    add("res-8", "restyle", "base", "give the selection square caps", "hand", "clock.svg")
    add("res-9", "restyle", "paraphrase", "Mitre the corners of this path", "star", "star.svg")
    add("res-10", "restyle", "base", "no stroke on the selected shape", "shadow", "ball.svg")

    # simplify - less detail, same subject.
    add("sim-1", "simplify", "base", "Simplify this icon", None, "castle.svg")
    add("sim-2", "simplify", "base", "Too many points, reduce the detail", None, "dragon.svg")
    add("sim-3", "simplify", "paraphrase", "Can you make this less busy?", None, "city.svg")
    add(
        "sim-4",
        "simplify",
        "paraphrase",
        "Flatten it down to the essential shapes",
        None,
        "tree.svg",
    )
    add(
        "sim-5",
        "simplify",
        "distractor",
        "Keep the colours but cut the fussy bits",
        None,
        "peacock.svg",
    )
    add(
        "sim-6",
        "simplify",
        "negation",
        "Don't redraw it, just strip out the small details",
        None,
        "map.svg",
    )
    add(
        "sim-7",
        "simplify",
        "base",
        "smooth the selected path and drop the tiny wiggles",
        "coastline",
        "island.svg",
    )
    add("sim-8", "simplify", "base", "make it cleaner, fewer curves", None, "cloud.svg")

    # clarify - missing target, missing value, or contradiction.
    add("ask-1", "clarify", "missing_target", "Change it", None, "hedgehog.svg")
    add("ask-2", "clarify", "missing_target", "Make it blue", None, "fish.svg")
    add("ask-3", "clarify", "missing_target", "Thicker", None, "fish.svg")
    add("ask-4", "clarify", "missing_target", "Fix the colour on that part", None, "car.svg")
    add(
        "ask-5",
        "clarify",
        "missing_target",
        "Make the selected shape a different colour",
        "body",
        "fish.svg",
    )
    add("ask-6", "clarify", "missing_target", "Change the stroke", "body", "fish.svg")
    add("ask-7", "clarify", "missing_target", "Make it nicer", None, "house.svg")
    add(
        "ask-8",
        "clarify",
        "missing_target",
        "Recolour the thing next to the other thing",
        None,
        "kitchen.svg",
    )
    add(
        "ask-9",
        "clarify",
        "missing_target",
        "Make it both red and green at the same time",
        "hat",
        "elf.svg",
    )
    add("ask-10", "clarify", "missing_target", "Do the usual", None, "logo.svg")
    add("ask-11", "clarify", "missing_target", "Edit the outline", None, "moon.svg")
    add(
        "ask-12",
        "clarify",
        "missing_target",
        "Make this one match the other one",
        "left_eye",
        "face.svg",
    )
    add("ask-13", "clarify", "missing_target", "colour", "body", "fish.svg")
    add("ask-14", "clarify", "missing_target", "Draw", None, None)

    assert len(rows) == 52, len(rows)
    return tuple(rows)


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy overall, per operation, per failure-mode kind, and the confusion matrix."""

    operations = [identifier for identifier, _ in OPERATIONS]
    confusion = {expected: {choice: 0 for choice in operations} for expected in operations}
    by_kind: dict[str, list[bool]] = {}
    by_operation: dict[str, list[bool]] = {}
    for row in results:
        hit = row["choice"] == row["expected"]
        confusion[row["expected"]][row["choice"]] += 1
        by_kind.setdefault(row["kind"], []).append(hit)
        by_operation.setdefault(row["expected"], []).append(hit)
    total = len(results)
    correct = sum(1 for row in results if row["choice"] == row["expected"])
    return {
        "decisions": total,
        "accuracy": correct / total if total else None,
        "per_operation": {
            key: sum(values) / len(values) for key, values in sorted(by_operation.items())
        },
        "per_kind": {key: sum(values) / len(values) for key, values in sorted(by_kind.items())},
        "confusion": confusion,
        "clarify_recall": by_operation.get("clarify")
        and sum(by_operation["clarify"]) / len(by_operation["clarify"]),
        "clarify_false_alarms": sum(
            1 for row in results if row["choice"] == "clarify" and row["expected"] != "clarify"
        ),
    }
