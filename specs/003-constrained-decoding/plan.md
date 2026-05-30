# Implementation Plan: Rule-Augmented Constrained Decoding Wrapper

**Branch**: `khaled_experiments` | **Date**: 2026-05-30 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/003-constrained-decoding/spec.md`

---

## Summary

Wrap the frozen Spec 001 autoregressive Transformer checkpoint with a HuggingFace `LogitsProcessor`-based layer that enforces the 10 process-logic rules from `generation_rules.md` at every decoding step — with no retraining. Hard-constraint mode masks illegal next tokens to `-inf`; soft-constraint mode adds configurable logit biases to preferred/discouraged tokens; both modes compose cleanly inside HF's `LogitsProcessorList`. The public API (`constrained_generate`) returns token sequences alongside rich per-step metadata (rules fired, masked tokens, deadlock flags) for downstream GUI and eval consumption.

---

## Technical Context

**Language/Version**: Python 3.11

**Primary Dependencies**:
- `torch >= 2.1` — inference engine (CPU-capable; A100 optional for speed)
- `transformers >= 4.35` — `LogitsProcessor`, `LogitsProcessorList`, `AutoModelForCausalLM`, `AutoTokenizer`
- `pydantic >= 2.0` — rule schema validation at load time
- `pyyaml >= 6.0` — YAML intermediate for rule graph serialisation
- `pytest >= 7.4` — test runner

**Storage**:
- Checkpoint read from `$WORK/checkpoints/spec001/` (Lustre; read-only at decode time)
- Rule file read from `tracks/industrial-infineon/training_data/generation_rules.md` (repo-local)
- No write paths required at inference time; eval outputs written to `results/003-constrained-decoding/`

**Testing**: `pytest` with `tests/decoding/` test directory; fixtures from held-out batches in `$WORK/data/fab_sequences/`

**Target Platform**: Leonardo login node (CPU path for API/unit tests); single A100 node via `boost_qos_dbg` for latency benchmarks

**Project Type**: Library (importable module + thin CLI eval script)

**Performance Goals**:
- `constrained_generate` latency within 10% of unconstrained for sequences up to 512 tokens on A100
- Rule violation rate = 0% under `mode="hard"` on 100-prompt benchmark
- `load_rules` completes in < 2 s on login node (pure Python, no GPU)

**Constraints**:
- No gradient computation; all inference in `torch.no_grad()`
- Multi-step lookahead capped at 3 steps to bound per-token overhead
- Must not modify the Spec 001 checkpoint or tokenizer artifacts
- `LogitsProcessor` subclass must be passable to `model.generate(logits_processor=[...])` without modification

**Scale/Scope**: ~200 vocab tokens, sequences up to 512 steps, 100-prompt benchmark, single-node inference

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| **Honest Evaluation** | MUST enforce | Eval script reports all four modes; no cherry-picking. Deadlock events logged and counted, not silently suppressed. |
| **Reproducibility** | MUST enforce | Rule loader is deterministic (sorted adjacency lists); random seed propagated through `gen_kwargs`; eval fixtures pinned with `SEED=42`. |
| **No retraining** | Compliant by design | `torch.no_grad()` throughout; `requires_grad=False` asserted in API init. |
| **Scope discipline** | Checked | GUI, training, LSTM baseline, and MLM variants explicitly out of scope. |

---

## Project Structure

### Documentation (this feature)

```text
specs/003-constrained-decoding/
├── spec.md
├── plan.md              ← this file
├── rule_schema.md       ← Phase 0 output: documented YAML/JSON schema (FR-007)
└── tasks.md             ← Phase 2 output (/speckit-tasks)
```

### Source Code

```text
src/
└── decoding/
    ├── __init__.py
    ├── rules.py          ← Rule loader, RuleGraph, Pydantic models, load_rules()
    ├── processors.py     ← ConstrainedLogitsProcessor (hard + soft)
    ├── api.py            ← constrained_generate() public API
    └── exceptions.py     ← RuleParseError, ConstraintDeadlockError, TokenizerMismatchError

tests/
└── decoding/
    ├── conftest.py       ← shared fixtures (tiny mock checkpoint, minimal rule set)
    ├── test_rules.py     ← Rule loader unit tests (all edge cases from spec)
    ├── test_processors.py← ConstrainedLogitsProcessor unit tests
    ├── test_api.py       ← constrained_generate integration tests
    └── test_eval.py      ← eval_constrained.py output validation

scripts/
└── eval_constrained.py  ← evaluation entry point (FR-012)
```

**Structure Decision**: Single-project library layout. The `src/decoding/` sub-package is importable by the Spec 001 training code and any future GUI without introducing additional top-level projects.

---

## Phase 0 — Rule Schema Design

**Goal**: Agree on the YAML/JSON intermediate schema before writing any code. This is the contract between the rule loader and the processors.

### Pydantic Models (`src/decoding/rules.py`)

```python
from pydantic import BaseModel, model_validator
from typing import Dict, List, Optional, Set, Tuple

class TransitionRule(BaseModel):
    rule_id: str                          # e.g. "RULE_DEP_NO_CLEAN"
    trigger_steps: List[str]              # steps that activate this rule
    required_preceding: List[str]         # any-of; must appear within lookahead window
    forbidden_preceding: List[str]        # none-of; must NOT appear within window
    window: int                           # lookback N (6, 12, or 15 per spec)
    variant_scope: Optional[List[str]]    # None = all families; else ["MOSFET","IC","IGBT"]
    soft_bias_delta: float = 0.0          # additive logit bias for preferred (positive) or
                                          # discouraged (negative) transitions; 0 = hard-only

class GlobalOrderRule(BaseModel):
    rule_id: str
    must_precede: str                     # token A must appear before token B
    required_before: str
    variant_scope: Optional[List[str]]

class LookaheadRule(BaseModel):
    rule_id: str
    trigger_steps: List[str]
    forbidden_suffix: List[str]           # sequence of steps that must NOT follow trigger
    depth: int                            # max 3

class CategoryExpansion(BaseModel):
    category: str                         # e.g. "DEPOSITION_*"
    members: List[str]                    # concrete step names in this category

class RuleGraph(BaseModel):
    forbidden_transitions: List[TransitionRule]
    preferred_transitions: List[TransitionRule]
    global_order_rules: List[GlobalOrderRule]
    lookahead_rules: List[LookaheadRule]
    category_expansions: Dict[str, List[str]]  # category -> member list
    variant_conditions: Dict[str, List[str]]   # rule_id -> applicable families

    # Derived (built post-validation, not serialised)
    _trigger_index: Dict[str, List[TransitionRule]] = {}   # step -> rules
    _global_index: Dict[str, List[GlobalOrderRule]] = {}   # required_before -> rules

    @model_validator(mode="after")
    def build_indexes(self) -> "RuleGraph":
        for r in self.forbidden_transitions + self.preferred_transitions:
            for step in r.trigger_steps:
                self._trigger_index.setdefault(step, []).append(r)
        for r in self.global_order_rules:
            self._global_index.setdefault(r.required_before, []).append(r)
        return self
```

### Schema YAML (stored in `specs/003-constrained-decoding/rule_schema.md`)

The 10 forbidden patterns from `generation_rules.md` map to Pydantic models as follows:

| Rule ID | Type | Window | Variant Scope |
|---|---|---|---|
| `RULE_DEP_NO_CLEAN` | `TransitionRule` (forbidden_preceding empty, required_preceding = clean set) | 12 | All |
| `RULE_METAL_ETCH_NO_LITHO` | `TransitionRule` | 15 | All |
| `RULE_ETCH_NO_MASK` | `TransitionRule` | 12 | All |
| `RULE_LITHO_LEVEL_SKIP` | `LookaheadRule` (prefix-tracking) | N/A | All |
| `RULE_IMPLANT_NO_MASK` | `TransitionRule` | 15 | All |
| `RULE_CMP_NO_DEP` | `TransitionRule` | 6 | All |
| `RULE_PAD_OPEN_BEFORE_DEP` | `GlobalOrderRule` | N/A | All |
| `RULE_TEST_BEFORE_PASSIVATION` | `GlobalOrderRule` | N/A | All |
| `RULE_SHIP_BEFORE_TEST` | `GlobalOrderRule` | N/A | All |
| `RULE_BACKSIDE_BEFORE_PASSIVATION` | `GlobalOrderRule` | N/A | All |

**Deliverable**: `specs/003-constrained-decoding/rule_schema.md` with full YAML example of each rule type.

---

## Phase 1 — Rule Loader + Tests

### `src/decoding/rules.py` — `load_rules(path: str) -> RuleGraph`

**Parsing strategy**: The loader does not regex-parse the entire markdown prose. Instead it reads the structured Section 3 table using a state machine keyed on the `### RULE_` headings, extracting:
1. Rule ID from the heading.
2. Trigger steps from the **Trigger steps:** field (parsed as a comma-separated inline list or bullet list).
3. Required/forbidden preceding from the **Required preceding:** field.
4. Window N from the `within N=` pattern in the description.
5. Variant scope from any `← MOSFET only` / `← IC only` annotations.

For global-order rules (RULE_PAD_OPEN_BEFORE_DEP, RULE_TEST_BEFORE_PASSIVATION, RULE_SHIP_BEFORE_TEST, RULE_BACKSIDE_BEFORE_PASSIVATION), the loader extracts the two ordered steps from the **Rule:** field.

**Error handling**:
- Unknown rule heading format → `RuleParseError` with line number (FR-014).
- Unknown step name (not in tokenizer vocab at validation time) → `UserWarning` with step name; rule is kept with the unknown step filtered out.
- Cyclic dependency in `GlobalOrderRule` must_precede graph → `RuleParseError` with the cycle path.
- Empty file → `RuleParseError`.

**Adjacency-list construction** (called after Pydantic validation):

```python
def build_adjacency(rule_graph: RuleGraph, vocab: Dict[str, int]) -> Dict[int, Set[int]]:
    """
    Returns forbidden_next[token_id] = set of token_ids that CANNOT follow token_id
    according to window-1 (immediate successor) hard rules.
    Full-window rules are evaluated at decode time against the prefix buffer.
    """
    forbidden_next: Dict[int, Set[int]] = defaultdict(set)
    for rule in rule_graph.forbidden_transitions:
        # Only trivially-immediate rules (window=1) go into the static adjacency list.
        # Window > 1 rules are checked at decode time via prefix scan.
        if rule.window == 1:
            for trigger in rule.trigger_steps:
                tid = vocab.get(trigger)
                if tid is None:
                    continue
                for bad_next in rule.forbidden_preceding:  # misleadingly named: bad successors
                    bid = vocab.get(bad_next)
                    if bid is not None:
                        forbidden_next[tid].add(bid)
    return dict(forbidden_next)
```

**Time complexity at sampling step**: The prefix scan is O(W × R) where W = max window (15) and R = number of rules (~10). With V ≈ 200 vocab tokens the mask construction is O(V) per step. Total per-step overhead is O(W × R + V) = O(1) relative to sequence length (all constants are small). Lookup of applicable rules by current token is O(1) via `_trigger_index`.

**Tests** (`tests/decoding/test_rules.py`):
- `test_load_all_10_rules` — parse real `generation_rules.md`; assert all 10 rule IDs present.
- `test_forbidden_transitions_schema` — assert each `TransitionRule` has non-empty trigger_steps and window > 0.
- `test_global_order_rules` — assert 4 global-order rules loaded with correct must_precede / required_before pairs.
- `test_category_expansion` — assert DEPOSITION_* category expands to the 19 deposition steps.
- `test_malformed_raises_parse_error` — pass a temp file with a malformed RULE_ heading; assert `RuleParseError` with line number.
- `test_empty_file_raises` — empty file → `RuleParseError`.
- `test_unknown_vocab_token_warns` — unknown step name emits `UserWarning`, does not raise.
- `test_cycle_detection` — synthetic rule graph with A must_precede B and B must_precede A → `RuleParseError` with cycle path.

---

## Phase 2 — Processors + Tests

### `src/decoding/processors.py`

#### HardConstraintProcessor

```python
from transformers import LogitsProcessor
import torch
from .rules import RuleGraph
from .exceptions import ConstraintDeadlockError

class HardConstraintProcessor(LogitsProcessor):
    """
    Masks logits of rule-violating next tokens to -inf.
    Stateful: maintains a prefix buffer per batch element.
    """

    def __init__(
        self,
        rule_graph: RuleGraph,
        tokenizer,
        variant: str,
        fallback: str = "temperature_backoff",  # or "raise"
        max_retries: int = 3,
    ):
        self.rule_graph = rule_graph
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()          # str -> int
        self.id_to_step = {v: k for k, v in self.vocab.items()}
        self.variant = variant
        self.fallback = fallback
        self.max_retries = max_retries
        self._prefix_buffers: dict[int, list[int]] = {}   # batch_idx -> token list
        self.metadata: list[list[dict]] = []              # per batch, per step

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape
        for batch_idx in range(batch_size):
            prefix = self._get_prefix(batch_idx, input_ids)
            illegal_ids, fired_rules = self._compute_illegal(prefix)

            masked_count = len(illegal_ids)
            scores[batch_idx, list(illegal_ids)] = float("-inf")

            # Deadlock detection
            conflict = False
            conflict_rules: list[str] = []
            if scores[batch_idx].max().item() == float("-inf"):
                conflict = True
                conflict_rules = fired_rules
                scores = self._apply_fallback(scores, batch_idx, fired_rules)

            # Record step metadata
            self._record_metadata(batch_idx, prefix, fired_rules, list(illegal_ids),
                                  conflict, conflict_rules)
        return scores

    def _compute_illegal(self, prefix: list[int]) -> tuple[set[int], list[str]]:
        """
        Scan prefix against all TransitionRules and GlobalOrderRules.
        Returns (illegal_token_id_set, fired_rule_id_list).

        Complexity: O(W * R + V) where W=max_window=15, R=10 rules, V=200 vocab.
        """
        illegal: set[int] = set()
        fired: list[str] = []

        for rule in self.rule_graph.forbidden_transitions:
            if rule.variant_scope and self.variant not in rule.variant_scope:
                continue
            window = prefix[-rule.window:] if len(prefix) >= rule.window else prefix
            window_steps = [self.id_to_step.get(t, "") for t in window]
            # Check: if trigger is current last step, required_preceding must appear in window
            if window_steps and window_steps[-1] in rule.trigger_steps:
                if not any(s in window_steps[:-1] for s in rule.required_preceding):
                    # The NEXT token triggers this rule's consequence:
                    # The next step after trigger must satisfy the rule.
                    # (rules constrain SUCCESSORS of trigger, not the trigger itself)
                    pass  # handled below via trigger_index
            # Prospective masking: for each candidate next token that IS a trigger,
            # check whether the window satisfies required_preceding.
            for trigger_step in rule.trigger_steps:
                trigger_id = self.vocab.get(trigger_step)
                if trigger_id is None:
                    continue
                window_steps_current = [self.id_to_step.get(t, "") for t in prefix[-rule.window:]]
                if not any(s in window_steps_current for s in rule.required_preceding):
                    illegal.add(trigger_id)
                    if rule.rule_id not in fired:
                        fired.append(rule.rule_id)

        for rule in self.rule_graph.global_order_rules:
            if rule.variant_scope and self.variant not in rule.variant_scope:
                continue
            prefix_steps = [self.id_to_step.get(t, "") for t in prefix]
            required_step_id = self.vocab.get(rule.required_before)
            must_precede_id = self.vocab.get(rule.must_precede)
            # If must_precede has NOT yet appeared, block required_before token now
            if required_step_id and must_precede_id:
                if rule.must_precede not in prefix_steps:
                    illegal.add(required_step_id)
                    if rule.rule_id not in fired:
                        fired.append(rule.rule_id)

        return illegal, fired

    def _apply_fallback(self, scores, batch_idx, fired_rules):
        if self.fallback == "raise":
            raise ConstraintDeadlockError(
                f"All tokens masked at step {len(self._prefix_buffers.get(batch_idx, []))}. "
                f"Conflicting rules: {fired_rules}"
            )
        # temperature_backoff: soften by multiplying scores by a large temperature
        # effectively un-masks everything and retries with soft-only
        scores[batch_idx] = scores[batch_idx] / 10.0  # temperature * 10
        scores[batch_idx][scores[batch_idx] == float("-inf")] = -1e4  # unblock
        return scores

    def _get_prefix(self, batch_idx: int, input_ids: torch.LongTensor) -> list[int]:
        return input_ids[batch_idx].tolist()

    def _record_metadata(self, batch_idx, prefix, fired_rules, masked_ids,
                         conflict, conflict_rules):
        while len(self.metadata) <= batch_idx:
            self.metadata.append([])
        self.metadata[batch_idx].append({
            "step_index": len(prefix),
            "mode": "hard",
            "fired_rules": fired_rules,
            "masked_token_ids": masked_ids,
            "biased_token_ids": [],
            "constraint_conflict": conflict,
            "conflicting_rule_ids": conflict_rules,
        })
```

#### SoftConstraintProcessor

```python
class SoftConstraintProcessor(LogitsProcessor):
    """
    Adds additive logit biases for preferred/discouraged transitions.
    Does NOT mask to -inf; hard illegality is not enforced here.
    """

    def __init__(self, rule_graph: RuleGraph, tokenizer, variant: str, soft_bias: float = 2.0):
        self.rule_graph = rule_graph
        self.vocab = tokenizer.get_vocab()
        self.id_to_step = {v: k for k, v in self.vocab.items()}
        self.variant = variant
        self.soft_bias = soft_bias
        self.metadata: list[list[dict]] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = scores.shape[0]
        for batch_idx in range(batch_size):
            prefix_steps = [self.id_to_step.get(t, "") for t in input_ids[batch_idx].tolist()]
            biased: list[tuple[int, float]] = []

            for rule in self.rule_graph.preferred_transitions:
                if rule.variant_scope and self.variant not in rule.variant_scope:
                    continue
                window_steps = prefix_steps[-rule.window:]
                for trigger_step in rule.trigger_steps:
                    tid = self.vocab.get(trigger_step)
                    if tid is None:
                        continue
                    # Positive bias for preferred successors
                    delta = rule.soft_bias_delta if rule.soft_bias_delta != 0.0 else self.soft_bias
                    scores[batch_idx, tid] += delta
                    biased.append((tid, delta))

            self._record_metadata(batch_idx, prefix_steps, biased)
        return scores

    def _record_metadata(self, batch_idx, prefix_steps, biased):
        while len(self.metadata) <= batch_idx:
            self.metadata.append([])
        self.metadata[batch_idx].append({
            "step_index": len(prefix_steps),
            "mode": "soft",
            "fired_rules": [str(b[0]) for b in biased],
            "masked_token_ids": [],
            "biased_token_ids": biased,
            "constraint_conflict": False,
            "conflicting_rule_ids": [],
        })
```

#### Composition via `LogitsProcessorList`

```python
from transformers import LogitsProcessorList

def build_processor_list(
    rule_graph: RuleGraph,
    tokenizer,
    variant: str,
    mode: str,             # "hard" | "soft" | "both" | "off"
    soft_bias: float = 2.0,
    fallback: str = "temperature_backoff",
) -> tuple[LogitsProcessorList, list]:
    """
    Returns (processor_list, [processor_refs_for_metadata_extraction]).
    In mode="both": hard runs first (applied first in list), then soft on legal tokens.
    """
    processors = []
    refs = []
    if mode in ("hard", "both"):
        h = HardConstraintProcessor(rule_graph, tokenizer, variant, fallback)
        processors.append(h)
        refs.append(h)
    if mode in ("soft", "both"):
        s = SoftConstraintProcessor(rule_graph, tokenizer, variant, soft_bias)
        processors.append(s)
        refs.append(s)
    return LogitsProcessorList(processors), refs
```

**Mode flag semantics**:
- `"hard"`: only `HardConstraintProcessor` in list → illegal tokens masked to `-inf`.
- `"soft"`: only `SoftConstraintProcessor` in list → biases applied, no masking.
- `"both"`: `Hard` first, then `Soft` — biases only affect the legally-reachable set (HF applies processors in list order, so hard masking precedes soft biasing at every step).
- `"off"`: empty list → standard `model.generate`.

**Soft bias tuning protocol**:
1. Run eval on a validation set with `mode="soft"` sweeping `soft_bias` ∈ {0.5, 1.0, 2.0, 3.0, 5.0}.
2. Plot rule-violation rate vs. perplexity delta.
3. Select the knee of the curve (maximum violation reduction with minimum perplexity degradation). Default `soft_bias=2.0` is a conservative starting point.

**Tests** (`tests/decoding/test_processors.py`):
- `test_hard_masks_illegal_token` — construct a prefix where RULE_SHIP_BEFORE_TEST fires; assert SHIP LOT token is -inf.
- `test_hard_mode_deterministic_single_legal` — prefix where only one token is legal; assert output is that token.
- `test_soft_adds_bias_not_masks` — assert no -inf values in soft mode even for forbidden transitions.
- `test_both_hard_before_soft` — verify hard masking precedes soft biasing by checking that bias is applied only to non-masked tokens.
- `test_off_mode_identity` — empty processor list passes scores unchanged.
- `test_deadlock_temperature_backoff` — synthetic all-forbidden vocab; assert metadata has `constraint_conflict=True`.
- `test_deadlock_raise_mode` — same setup with `fallback="raise"`; assert `ConstraintDeadlockError`.
- `test_metadata_schema` — assert returned metadata dicts contain all keys from FR-009.
- `test_empty_rules_behaves_like_off` — RuleGraph with no rules; assert hard mode = off mode output.

---

## Phase 3 — Public API + Evaluation

### `src/decoding/api.py`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .rules import RuleGraph, load_rules
from .processors import build_processor_list
from .exceptions import TokenizerMismatchError
from typing import Any

def constrained_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str | list[int],
    rules: RuleGraph,
    mode: str = "hard",               # "hard" | "soft" | "both" | "off"
    variant: str = "IC",
    soft_bias: float = 2.0,
    fallback: str = "temperature_backoff",
    **gen_kwargs: Any,
) -> tuple[list[list[int]], list[list[dict]]]:
    """
    Run constrained autoregressive generation.

    Returns
    -------
    sequences : List[List[int]]
        Generated token-ID sequences (one per beam/sample).
    metadata  : List[List[dict]]
        Parallel list. metadata[i][t] is the StepMetadata dict for
        sequence i at decoding step t (see FR-009).
    """
    # Tokenizer / model embedding size check (FR-010)
    vocab_size = len(tokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if vocab_size != embedding_size:
        raise TokenizerMismatchError(
            f"Tokenizer vocab size {vocab_size} != model embedding size {embedding_size}. "
            "Ensure the same tokenizer artifact used during Spec 001 training is passed."
        )

    if isinstance(prompt, str):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
    else:
        input_ids = torch.tensor([prompt])

    processor_list, processor_refs = build_processor_list(
        rules, tokenizer, variant, mode, soft_bias, fallback
    )

    with torch.no_grad():
        output = model.generate(
            input_ids,
            logits_processor=processor_list,
            **gen_kwargs,
        )

    sequences = output.tolist()

    # Merge metadata from all processors (hard + soft)
    # Both processors accumulate per-step metadata independently;
    # merge by step_index, combining masked_token_ids and biased_token_ids.
    merged_metadata = _merge_processor_metadata(processor_refs, sequences)

    return sequences, merged_metadata


def _merge_processor_metadata(
    processor_refs: list,
    sequences: list[list[int]],
) -> list[list[dict]]:
    """Merge StepMetadata dicts from hard and soft processors by step_index."""
    if not processor_refs:
        # mode="off": return empty metadata per step per sequence
        return [[{"step_index": i, "mode": "off", "fired_rules": [],
                  "masked_token_ids": [], "biased_token_ids": [],
                  "constraint_conflict": False, "conflicting_rule_ids": []}
                 for i in range(len(seq))] for seq in sequences]

    # Collect all metadata lists (one list per processor, per batch element)
    n_seqs = len(sequences)
    merged: list[list[dict]] = [[] for _ in range(n_seqs)]
    for batch_idx in range(n_seqs):
        steps_meta: dict[int, dict] = {}
        for proc in processor_refs:
            if batch_idx >= len(proc.metadata):
                continue
            for step_dict in proc.metadata[batch_idx]:
                si = step_dict["step_index"]
                if si not in steps_meta:
                    steps_meta[si] = {
                        "step_index": si,
                        "mode": step_dict["mode"],
                        "fired_rules": [],
                        "masked_token_ids": [],
                        "biased_token_ids": [],
                        "constraint_conflict": False,
                        "conflicting_rule_ids": [],
                    }
                steps_meta[si]["fired_rules"] += step_dict["fired_rules"]
                steps_meta[si]["masked_token_ids"] += step_dict["masked_token_ids"]
                steps_meta[si]["biased_token_ids"] += step_dict["biased_token_ids"]
                if step_dict["constraint_conflict"]:
                    steps_meta[si]["constraint_conflict"] = True
                    steps_meta[si]["conflicting_rule_ids"] += step_dict["conflicting_rule_ids"]
        merged[batch_idx] = [steps_meta[i] for i in sorted(steps_meta)]
    return merged
```

### `src/decoding/exceptions.py`

```python
class RuleParseError(Exception):
    """Raised by load_rules() for structurally invalid input. Includes line number."""
    pass

class ConstraintDeadlockError(Exception):
    """Raised when all tokens are masked and fallback='raise'."""
    pass

class TokenizerMismatchError(Exception):
    """Raised when tokenizer vocab size != model embedding dimension."""
    pass
```

### `scripts/eval_constrained.py` (FR-012)

**Metrics reported** (all four modes, for each of 100 benchmark prompts):
1. **Rule violation rate** — `validate_sequence(steps)` from `generate_sequences.py`; count violations / total sequences.
2. **Perplexity / NLL delta** — compute token NLL under frozen model; report delta vs `mode="off"`.
3. **Mean latency overhead %** — `(wall_clock_constrained - wall_clock_off) / wall_clock_off * 100`; averaged over all prompts.

**Output**: `results/003-constrained-decoding/eval_results.json` with all metrics per mode; also printed as a Markdown table to stdout.

**Test** (`tests/decoding/test_eval.py`): Run eval script on a 5-prompt mini-fixture; assert the JSON output contains all required keys and hard-mode violation rate == 0.

---

## Deadlock Handling

When `HardConstraintProcessor` detects that `scores[batch_idx].max() == -inf` after masking:

**`fallback="temperature_backoff"` (default)**:
1. Divide all scores by 10 (equivalent to temperature × 10 — flattens distribution).
2. Replace any remaining `-inf` values with `-1e4` (unblocks the full vocabulary).
3. Retry up to `max_retries=3` times. If still deadlocked after 3 retries, raise `ConstraintDeadlockError`.
4. Record `constraint_conflict=True` and all `conflicting_rule_ids` in step metadata.

**`fallback="raise"`**:
1. Raise `ConstraintDeadlockError` immediately with conflicting rule IDs.
2. Still record the conflict in metadata before raising.

**Expected deadlock frequency**: Under the 10-rule set and the current grammar, deadlocks should be rare (< 1% of steps) for well-formed prompts. If deadlock frequency exceeds 5% on the benchmark, this is a signal of a rule-parsing bug (rules being over-applied) — investigate `_compute_illegal` logic first.

---

## Evaluation Protocol

### Benchmark Setup

- **Prompt set**: 100 prompts sampled from `eval_input_valid.csv` (held-out; pipe-separated partial sequences at 60% completion fraction).
- **Family split**: ~33 per family (MOSFET, IGBT, IC); variant token passed as `variant` arg.
- **Sequence length**: complete to 512 tokens max.
- **Hardware**: login node (CPU) for mode comparison; `boost_qos_dbg` for latency measurement on A100.

### Measurement Protocol

```
for mode in ["off", "hard", "soft", "both"]:
    t0 = time.perf_counter()
    seqs, meta = constrained_generate(model, tokenizer, prompt, rules, mode=mode,
                                      max_new_tokens=200, do_sample=True, temperature=0.8)
    t1 = time.perf_counter()
    violations = sum(len(validate_sequence(decode(s))) > 0 for s in seqs)
    nll = compute_nll(model, seqs)
    latency = (t1 - t0)
```

**Latency overhead** = `(latency_hard - latency_off) / latency_off`. Target: < 10%.

**Success gate**: After Phase 3 eval, if latency overhead > 10%, profile `_compute_illegal` and optimise the prefix scan (candidate: pre-compute which rules fire for each trigger token at load time, replacing the inner loop with a direct lookup).

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Rule parsing ambiguity** — markdown prose in `generation_rules.md` uses inconsistent list formats; the state-machine parser may miss steps. | Medium | High (silent incorrect rule coverage) | Validate parsed rule graph against a hand-coded ground-truth fixture (all 10 rule IDs, key trigger steps, windows). Add a `--dry-run` flag to `eval_constrained.py` that prints the parsed rule graph for human inspection. |
| **Latency budget breach** — `_compute_illegal` scans prefix × rules at every token; for long sequences and large batches this may exceed 10%. | Low-Medium | Medium (degrades demo speed) | Pre-compute per-trigger-token rule lookups at `load_rules` time. Cache the illegal-token set for the last N=15 tokens; invalidate on prefix change. |
| **Deadlock frequency** — over-broad rule parsing (e.g., wrong window or wrong trigger set) causes frequent deadlocks, crashing the demo. | Low | High (breaks demo) | Unit tests with known-valid prefixes assert 0 deadlocks. Integration test with 20 randomly generated valid sequences asserts deadlock_count == 0. |
| **Tokenizer mismatch** — Spec 001 checkpoint trained with a custom step-level tokenizer; if `AutoTokenizer` is not the exact same artifact, vocab sizes diverge silently. | Medium | Critical (wrong masks applied) | `TokenizerMismatchError` raised at API init (FR-010). Document the exact tokenizer path in the API docstring. |
| **Variant scope gaps** — rules with no variant_scope annotation are applied to all families; an IC-specific rule incorrectly masking MOSFET tokens would lower top-1 accuracy. | Low | Medium | Per-family violation rate reported separately in eval; alert if one family has significantly higher NLL delta. |

---

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No constitution violations. The stateful `_prefix_buffers` design adds complexity but is unavoidable: window-based rules require prefix history, which HF's stateless `LogitsProcessor` interface does not natively provide. The workaround (reading from `input_ids` passed at each call) is idiomatic for HF and does not require an additional abstraction layer.

---

## Phase Plan

| Phase | Deliverables | Owner | Target |
|---|---|---|---|
| **Phase 0** — Rule Schema Design | `rule_schema.md`; Pydantic model definitions committed to `src/decoding/rules.py`; all 10 rules hand-mapped to schema types | Any | Before Phase 1 starts |
| **Phase 1** — Loader + Tests | `load_rules()` implementation; `test_rules.py` all 8 tests green; `generate_sequences.validate_sequence` smoke-tested against parsed rules | Any | Login node; no GPU needed |
| **Phase 2** — Processors + Tests | `HardConstraintProcessor`, `SoftConstraintProcessor`, `build_processor_list`; `test_processors.py` all 9 tests green; exceptions module complete | Any | Login node (CPU inference with tiny mock model) |
| **Phase 3** — Integration Eval | `constrained_generate` API; `eval_constrained.py`; all success criteria measured; `test_api.py` + `test_eval.py` green; eval results committed to `results/003-constrained-decoding/` | Any | `boost_qos_dbg` (A100) for latency; login node for correctness |

**Critical path**: Phase 0 → Phase 1 → Phase 2 → Phase 3. Phase 1 and Phase 2 can be parallelised across teammates once Phase 0 schema is agreed.

---

## Open Questions

1. **Tokenizer artifact location**: The spec assumes the Spec 001 tokenizer is at `$WORK/checkpoints/spec001/`. If the Spec 001 team stores it elsewhere (e.g., a separate `tokenizer/` directory), the API docstring and `TokenizerMismatchError` message need updating. Confirm with Spec 001 owner before Phase 1.

2. **`RULE_LITHO_LEVEL_SKIP` parsing**: This rule requires tracking the sequence of `ALIGN MASK LEVEL N` tokens and asserting monotonic ordering. Unlike the other 9 rules it is not a simple window lookup — it requires a global counter over the prefix. Recommend implementing it as a special-cased check in `_compute_illegal` rather than fitting it into `LookaheadRule`. Confirm approach in Phase 0 review.

3. **Soft bias per-rule vs. global scalar**: The spec allows `soft_bias_delta` per rule (encoded in `TransitionRule`) but defaults to the global `soft_bias` kwarg. If the tuning sweep (Phase 3) reveals that different rules need different magnitudes, the per-rule field is already available. No schema change needed.

4. **Batch size > 1 with variant mixing**: The current API takes a single `variant` string. If a caller wants to generate IC and MOSFET sequences in the same batch, they must call `constrained_generate` twice. This is by design (variant-scoped rules are computed once at processor init). Document this limitation in the API docstring.
