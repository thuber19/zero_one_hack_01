# Feature Specification: Rule-Augmented Constrained Decoding Wrapper

**Feature Branch**: `003-constrained-decoding`

**Created**: 2026-05-30

**Status**: Draft

**Input**: User description: "Decoding-time wrapper around the trained autoregressive Transformer (Spec 001) that combines hard constraints (mask illegal next-step tokens to -inf in logits) with soft constraints (logit biasing for preferred orderings). No retraining — operates on a frozen checkpoint."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Constrained Generation API (Priority: P1)

A researcher calls `constrained_generate(model, tokenizer, prompt, rules, mode="hard", ...)` and receives generated fab-process sequences together with per-step constraint metadata showing exactly which rules fired and which tokens were masked at every decoding step.

**Why this priority**: This is the core deliverable. Without the API, nothing else in the feature works or can be evaluated. It directly enables the GUI integration and the hackathon demo.

**Independent Test**: Load the Spec 001 checkpoint, call `constrained_generate` with a minimal rule set and `mode="hard"`, verify the returned sequences contain no rule-violating transitions and that the metadata dict contains `masked_tokens` and `fired_rules` keys at each step.

**Acceptance Scenarios**:

1. **Given** a frozen Spec 001 checkpoint and a non-empty rule graph, **When** `constrained_generate` is called with `mode="hard"`, **Then** it returns a list of token-ID sequences and a parallel list of per-step metadata dicts containing `step_index`, `fired_rules`, and `masked_token_ids`.
2. **Given** a prompt whose next legal tokens are fully constrained to a single option, **When** `mode="hard"` is active, **Then** the model deterministically picks that token regardless of its raw logit value.
3. **Given** `mode="soft"`, **When** `constrained_generate` is called, **Then** logit biases are added (not hard masked) and the metadata reports `biased_token_ids` with the applied delta per token.
4. **Given** `mode="both"`, **When** called, **Then** hard masking is applied first and soft biasing is applied to the remaining legal tokens.
5. **Given** `mode="off"`, **When** called, **Then** output is identical to calling the model's standard `generate` with the same `gen_kwargs`.

---

### User Story 2 - Rule Violation Rate Drops vs Unconstrained (Priority: P1)

A evaluator runs the provided benchmark script and sees that the rule violation rate under `mode="hard"` is 0% (all violations masked) and under `mode="soft"` is significantly lower than unconstrained, while latency overhead over unconstrained sampling stays below 10%.

**Why this priority**: This is the primary correctness and performance contract. Without measurable improvement the feature has no value over plain decoding.

**Independent Test**: Run `eval_constrained.py` on a held-out set of 100 process-flow prompts; compare violation counts and wall-clock time between `mode="off"` and `mode="hard"`.

**Acceptance Scenarios**:

1. **Given** a 100-prompt benchmark set and the full rule graph, **When** evaluated with `mode="hard"`, **Then** rule violation rate is 0%.
2. **Given** the same benchmark, **When** evaluated with `mode="soft"`, **Then** rule violation rate is at least 50% lower than `mode="off"`.
3. **Given** `mode="hard"` running on a single A100 or CPU (API path), **When** generating sequences of up to 512 tokens, **Then** wall-clock latency per sequence is within 10% of unconstrained generation.

---

### User Story 3 - Rule Loader Parses generation_rules.md (Priority: P2)

A developer calls `load_rules("tracks/industrial-infineon/training_data/generation_rules.md")` and receives a validated, structured rule graph (YAML/JSON intermediate) that the constrained decoder can consume directly.

**Why this priority**: The rule loader is a prerequisite for the API but can be developed and tested independently against the markdown file. Having it as a separate story lets a second teammate work on it in parallel.

**Independent Test**: Parse `generation_rules.md` with `load_rules`, print the resulting data structure, and assert it contains at minimum the keys `forbidden_transitions`, `preferred_transitions`, and `variant_conditions` with correct types.

**Acceptance Scenarios**:

1. **Given** `generation_rules.md` exists at the expected path, **When** `load_rules` is called, **Then** it returns a `RuleGraph` object (or dict conforming to the documented schema) without raising exceptions.
2. **Given** a rule referencing a step category (e.g., `"DEPOSITION_*"`), **When** loaded, **Then** the rule graph expands or tags it so the decoder can match against specific vocab tokens that belong to that category.
3. **Given** a variant-conditional rule (e.g., "only applies to 65 nm process"), **When** loaded, **Then** the rule graph carries the variant predicate and the decoder can evaluate it against runtime context.
4. **Given** a malformed or empty `generation_rules.md`, **When** `load_rules` is called, **Then** it raises a descriptive `RuleParseError` (not a bare exception) with the offending line number.

---

### User Story 4 - Failure Handling When No Legal Token Remains (Priority: P3)

When hard constraints leave zero legal next tokens (all logits masked to -inf), the system does not crash silently; instead, it applies a documented fallback strategy and surfaces a warning in the metadata.

**Why this priority**: Edge case resilience. The P1 stories deliver the happy path; this protects the demo from surprising failures.

**Independent Test**: Construct an artificial rule set that forbids every token in the vocabulary given a specific prefix, call `constrained_generate` with `mode="hard"`, and verify the fallback triggers and the returned metadata includes `constraint_conflict=True`.

**Acceptance Scenarios**:

1. **Given** a prefix where all tokens are forbidden by hard rules, **When** `constrained_generate` is called with `fallback="temperature_backoff"` (default), **Then** the decoder increases temperature and retries up to 3 times before raising `ConstraintDeadlockError`.
2. **Given** the same scenario with `fallback="raise"`, **When** called, **Then** `ConstraintDeadlockError` is raised immediately with a message listing the conflicting rule IDs.
3. **Given** any deadlock regardless of fallback, **When** the call returns or raises, **Then** the per-step metadata for the conflicting step includes `constraint_conflict=True` and `conflicting_rule_ids`.

---

### Edge Cases

- What happens when the rule set is empty? `constrained_generate` with `mode="hard"` or `mode="soft"` should behave identically to `mode="off"` and log a warning.
- What happens when hard rules conflict with each other (cyclic references that make every next step illegal from the very first token)? `load_rules` should detect cycles and raise `RuleParseError` with the cycle path.
- What happens when the tokenizer loaded at decode time does not match the one used during training? A vocab-size mismatch check should raise `TokenizerMismatchError` before any decoding begins.
- What happens when a rule references a step name that does not exist in the tokenizer vocabulary? `load_rules` should warn (not raise) and skip the unknown token, logging its name.
- What happens with multi-step lookahead rules when the sequence reaches EOS before the lookahead horizon? Lookahead checking is truncated at EOS; no constraint violation is recorded.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST expose a public function `constrained_generate(model, tokenizer, prompt, rules, mode, **gen_kwargs)` returning `(sequences: List[List[int]], metadata: List[List[dict]])`.
- **FR-002**: `mode` MUST accept exactly four string values: `"hard"`, `"soft"`, `"both"`, `"off"`.
- **FR-003**: In `mode="hard"`, system MUST set logits of all rule-violating next tokens to `-inf` before sampling at every decoding step.
- **FR-004**: In `mode="soft"`, system MUST add a configurable additive bias (default magnitude tunable via `soft_bias` kwarg, suggested range -5.0 to +5.0) to preferred/discouraged token logits; hard illegal tokens are NOT masked.
- **FR-005**: In `mode="both"`, hard masking MUST be applied before soft biasing so biases only affect the legally-reachable token set.
- **FR-006**: The wrapper MUST implement the HuggingFace `LogitsProcessor` interface so it can be passed as a `logits_processor` argument to any HF `generate()` call.
- **FR-007**: System MUST expose `load_rules(path: str) -> RuleGraph` that parses `generation_rules.md` into a validated intermediate structure (YAML/JSON schema documented in `specs/003-constrained-decoding/rule_schema.md`).
- **FR-008**: `RuleGraph` MUST support: forbidden transitions (step A → step B is illegal), preferred transitions (logit bias up), step-category wildcards, variant-conditional rules, and multi-step lookahead rules (up to 3 steps).
- **FR-009**: Per-step metadata dict MUST contain: `step_index` (int), `mode` (str), `fired_rules` (List[str] — rule IDs), `masked_token_ids` (List[int], empty if mode != "hard"/"both"), `biased_token_ids` (List[Tuple[int, float]], empty if mode != "soft"/"both"), `constraint_conflict` (bool), `conflicting_rule_ids` (List[str]).
- **FR-010**: System MUST check tokenizer vocabulary size against model embedding size at initialization and raise `TokenizerMismatchError` if they differ.
- **FR-011**: When hard constraints leave no legal token, system MUST apply fallback strategy (`"temperature_backoff"` or `"raise"`) and record `constraint_conflict=True` in metadata.
- **FR-012**: The evaluation script `eval_constrained.py` MUST report: rule violation rate, perplexity/NLL (vs unconstrained), and mean latency overhead percentage, for all four modes.
- **FR-013**: System MUST NOT require any gradient computation or parameter updates — the frozen checkpoint is used in `torch.no_grad()` inference mode throughout.
- **FR-014**: `load_rules` MUST raise `RuleParseError` (with line number) for malformed input and emit a `UserWarning` (not raise) for rules that reference unknown vocabulary tokens.

### Key Entities

- **RuleGraph**: The parsed, validated representation of all constraints. Contains `forbidden_transitions`, `preferred_transitions`, `variant_conditions`, `lookahead_rules`, and `category_expansions`. Immutable after construction.
- **ConstrainedLogitsProcessor**: HuggingFace `LogitsProcessor` subclass. Holds a reference to `RuleGraph`, the current `mode`, and `soft_bias` magnitude. Stateful per-sequence: tracks the decoded prefix to evaluate context-dependent rules.
- **StepMetadata**: Per-decoding-step dict (see FR-009). Accumulated into a list and returned alongside token sequences.
- **RuleParseError**: Custom exception raised by `load_rules` on structurally invalid input.
- **ConstraintDeadlockError**: Custom exception raised when all tokens are masked and `fallback="raise"`.
- **TokenizerMismatchError**: Custom exception raised when tokenizer vocab size != model embedding dim.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `constrained_generate` with `mode="hard"` produces 0% rule violations on the 100-prompt benchmark set drawn from the Infineon process-flow dataset.
- **SC-002**: `constrained_generate` with `mode="hard"` adds no more than 10% wall-clock latency overhead versus `mode="off"` on a single A100 GPU for sequences up to 512 tokens.
- **SC-003**: `constrained_generate` with `mode="soft"` reduces rule violation rate by at least 50% relative to `mode="off"` on the same benchmark.
- **SC-004**: `load_rules` parses `generation_rules.md` in under 2 seconds on the login node (pure Python, no GPU).
- **SC-005**: The `ConstrainedLogitsProcessor` passes HuggingFace's `LogitsProcessor` interface checks (can be supplied to `model.generate(logits_processor=[...])` without modification).
- **SC-006**: All four edge-case scenarios in the spec (empty rules, cyclic rules, tokenizer mismatch, unknown vocab token) are covered by automated tests that pass.

## Assumptions

- The Spec 001 autoregressive Transformer checkpoint is available at `$WORK/checkpoints/spec001/` and is loadable via `transformers.AutoModelForCausalLM.from_pretrained` or equivalent.
- The same tokenizer artifact used during Spec 001 training is available alongside the checkpoint and will be passed to `constrained_generate` by the caller.
- `generation_rules.md` is the authoritative, human-maintained source of process rules; the rule loader treats it as read-only input.
- Soft bias magnitudes do not need to be learned; the caller sets `soft_bias` as a scalar kwarg. Automatic magnitude tuning is out of scope.
- The HuggingFace `transformers` library (>= 4.35) is available in the project conda environment at `$WORK/envs/`.
- Multi-step lookahead is limited to 3 steps to keep per-token overhead bounded; deeper lookahead is out of scope.
- The GUI integration (Spec 002 or equivalent) consumes the returned metadata but is not built in this feature.
- Training, MLM variants, LSTM baselines, and GUI code are explicitly out of scope.
- Evaluation runs on `boost_qos_dbg` (30-min limit) or the login node (CPU path); no full production SLURM job is required for this spec.
