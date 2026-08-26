# Contributing to saroku

## Contributing probes to the behavioral safety benchmark

This is the most valuable and most sensitive kind of contribution saroku accepts, so
this section is long on purpose. Read it before opening a submission.

### Why this process exists

`bench_v1` (`src/saroku/benchmarks/bench_v1.py`) is 96 hand-authored probe instances
across saroku's 8 behavioral properties — sycophancy, honesty, consistency,
prompt_injection, trust_hierarchy, corrigibility, minimal_footprint, goal_drift. It's
saroku's core IP: small, hand-authored, immutable, versioned, and citable, in the same
spirit as benchmarks like HumanEval. **It is never auto-generated and never edited in
place** — see `CLAUDE.md`.

That immutability is a feature, not a limitation, but it also means `bench_v1` alone —
about 12 probes per property — is too thin to support confident, statistically
meaningful claims about detection rates for any one property. Growing real coverage
means adding a **separate, additive** pool of probes, not diluting or mutating the
original. That's what this process is for.

The single most valuable thing you can contribute is a probe based on **a real failure
you actually observed** — an agent that caved under pressure, executed something it
shouldn't have, or resisted a correction it should have accepted. Real, non-templated
failure scenarios are exactly the kind of data that's hardest to synthesize and most
valuable to have. Hypothetical-but-well-constructed probes are welcome too — provenance
is encouraged, not required.

### How to submit

Open a GitHub issue using the **Probe Submission** template
(`.github/ISSUE_TEMPLATE/probe-submission.md`). It asks for:

- Which of the 8 behavioral properties this tests
- Domain (e.g. finance, healthcare, DevOps, customer support)
- The scenario: what the agent is asked to do, and under what pressure
- What the correct/safe behavior actually is, and why
- The pressure variants — the specific phrasing used to try to talk the agent out of
  the correct behavior (authority claims, urgency, false consensus, etc.)
- *(Optional but encouraged)* What actually happened — a real incident, a real model
  failure you saw, with any identifying details redacted

A PR that directly adds a `ProbeInstance`-shaped submission is also fine if you're
comfortable with the schema (see `src/saroku/core/schema.py`) — but the issue template
is the lower-friction path and doesn't require writing Python.

### What happens after you submit

1. **A maintainer triages it.** This is not automatic and not fast — see the anti-abuse
   note below for why.
2. **Automated checks run** (`scripts/validate_community_probe.py`):
   - Schema/completeness check — does the submission have every field required for
     its stated property (see `REQUIRED_FIELDS` in the script)?
   - **Deduplication against `bench_v1`** — a lexical 5-gram overlap check against
     every field of all 96 real `bench_v1` instances. This exists specifically to
     protect `bench_v1`'s integrity as a clean, uncontaminated benchmark — if your
     submission is judged too similar to an existing instance, it'll be flagged for
     you to revise, not silently rejected.
   - Once a batch of accepted submissions for a property reaches meaningful volume, a
     trivial-classifier overfitting check runs across the batch — if a dumb
     TF-IDF+logistic-regression model can solve the batch with >90% accuracy, that's a
     sign the batch has learned a surface-level shortcut instead of genuine diversity,
     and the batch gets revisited before release.
3. **A maintainer independently verifies the stated `correct_answer` is actually
   correct.** Automated checks passing is not approval — see anti-abuse below for why
   this step is non-negotiable.
4. **Approved submissions move to `src/saroku/benchmarks/community_staging/`** — a
   holding pool, explicitly *not* part of any scored or citable benchmark yet (see that
   package's docstring). Staged entries carry contributor attribution and, where given,
   the real-world source note.
5. **Periodically, a maintainer curates a batch of staged, approved entries into a new
   versioned release** (e.g. `bench_v2.py`), following the exact same philosophy as
   `bench_v1`: fixed, immutable once released, reproducible, citable. `bench_v1` itself
   is never touched by this process.

### Anti-abuse — read this if you're wondering why review is required

A public benchmark for a safety product is a plausible target for bad-faith
submissions: someone could submit a probe with a deliberately mislabeled
`correct_answer` — for example, describing an actually-unsafe action as the "safe"
choice — hoping it gets trusted uncritically and quietly corrupts the benchmark or
anything trained/evaluated against it later. This is exactly why:

- There is no auto-merge path, ever, regardless of how clean the automated checks come
  back.
- A maintainer must independently reason about and confirm correctness, not just check
  that the submission is well-formed.
- Attribution is kept (who submitted what), so a pattern of bad submissions from one
  source is traceable.

If you're a maintainer reviewing a submission: treat "the automated checks passed" as
"this is safe to spend review time on," never as "this is safe to accept."

### Attribution

Contributors are credited by name/handle in `community_staging` entries and in the
changelog/release notes of whichever versioned release (e.g. `bench_v2`) eventually
includes their probe. If you'd rather not be credited, say so in your submission.

### What this unlocks

Once the community pool has real volume, it becomes the mechanism for reporting
statistically meaningful per-property numbers — pairing `bench_v1`'s small, trusted
core with enough additional volume to say more than "N out of 12." No specific volume
target or timeline is promised; this grows as real contributions come in.

---

## Other contributions

Bug reports, framework integration improvements, adapter additions for new LLM
providers, and documentation fixes are all welcome via standard GitHub issues/PRs. If
you're touching `src/saroku/policy/dsl.py` or adding fields to the policy schema, read
the governance rule in that file's module docstring first — the policy layer is
deliberately kept to flat, declarative YAML with no custom expression syntax, by
design, not by oversight.
