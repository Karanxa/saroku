# corrigibility_v1 — data card

**Status: DRY RUN ONLY. Not fit to train on.** The quantitative gate (safeguard #4)
correctly caught this and blocked promotion to a real training set — see below.

## Generation method
- Source: `Anthropic/model-written-evals`, `advanced-ai-risk/human_generated_evals/corrigible-{less,neutral,more}-HHH.jsonl` (CC-BY-4.0). 1,000 source disposition-questions, core instruction gist extracted (multiple-choice scaffolding stripped).
- Pipeline: `/home/karan/saroku/data_generation/generate_corrigibility.py`, backend=`template` (combinatorial phrase-bank rendering — **no LLM calls made**).
- Blocker: no `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` found in `/home/karan/saroku-demo/.env` (only `XAI_API_KEY` — unfunded per this session's earlier diagnosis — and `GOOGLE_API_KEY` — depleted credits, confirmed same session). The `llm_paraphrase()` backend is implemented and ready but **has never actually run**; the fallback `template` backend produced this file instead, purely to validate pipeline mechanics end-to-end.
- Six safeguards implemented in code: axis rotation (domain × framing × action-style × context-style), minimal pairs, shared action-function vocabulary, n-gram + trivial-classifier gates, disjoint held-out generation seeds, blended tool-report/conversational context styles.

## Counts
- Train: 700 (350 corrigibility / 350 safe minimal pairs)
- Held-out: 100 (50/50), drawn from 2 domains and 1 framing never used in training

## Gate results (this is the important part)
- **N-gram audit: FAILED.** Multiple phrases appear in >36% of examples per label (e.g. "as part of its assigned responsibilities" in 36% of corrigibility examples, "the agent stops" in 37% of safe examples) — far above the 15-20% threshold.
- **Trivial classifier probe: FAILED.** TF-IDF + logistic regression scores **100% accuracy on both train and held-out**. This is a hard confirmation that the template phrase banks are exactly the kind of narrow, learnable surface pattern the user explicitly asked us to avoid — a fixed vocabulary of ~8-10 phrase-bank entries per slot is trivially separable no matter how many axes are combinatorially rotated around it.
- **bench_v1 overlap: PASSED.** Max lexical 5-gram Jaccard overlap = 0.0 against all 47 string literals in `bench_v1.py`'s `CORRIGIBILITY_INSTANCES` block. Zero contamination risk from this file.

## Conclusion
The pipeline's own gates worked exactly as designed: they caught that a template/phrase-bank generator — even with axis rotation, minimal pairs, and shared vocabulary — is not sufficient diversity on its own. Combinatorial rotation over a small fixed phrase bank is still a template; the trivial-classifier probe proves it. **Real linguistic diversity requires the LLM paraphrase pass** (`--backend llm`, code already written in `llm_paraphrase()`), which needs a working `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

## Next step
Supply one of those keys, rerun `python generate_corrigibility.py --backend llm --pairs 350`, and re-check the same three gates. Do not train on `corrigibility_v1.jsonl` as it stands — it is a mechanics smoke-test artifact, not training data.
