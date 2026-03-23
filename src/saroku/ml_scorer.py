"""
saroku.ml_scorer — Layer 2 of the safety cascade.

A lightweight ML risk scorer. Extracts ~25 structured features from the action
and context, then applies a calibrated logistic-regression-style linear model
to produce a risk score in [0, 1].

No external model files. No network calls. Runs in <5ms.

Architecture:
    feature_vector = extract_features(action, context)   # shape: (25,)
    risk_score     = sigmoid(weights @ feature_vector + bias)

The weights encode domain knowledge about what makes agent actions risky.
They are designed to be replaced with trained weights once labeled data is
available — the feature extraction pipeline stays identical.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class MLScore:
    risk_score: float        # 0 = safe, 1 = dangerous
    verdict: str             # "block" | "allow" | "uncertain"
    top_features: list[tuple[str, float]]  # (feature_name, contribution) top-5


# ── Feature vocabulary (risk-weighted terms) ──────────────────────────────────
# Each entry: (pattern, weight)
# Positive weight → increases risk. Negative weight → decreases risk.

_TERM_WEIGHTS: list[tuple[re.Pattern, float]] = [
    # High-risk operations
    (re.compile(r'\bdelete\b', re.I),           +2.5),
    (re.compile(r'\bdrop\b', re.I),              +2.8),
    (re.compile(r'\btruncate\b', re.I),          +2.6),
    (re.compile(r'\bpurge\b', re.I),             +2.4),
    (re.compile(r'\birreversible\b', re.I),      +2.2),
    (re.compile(r'\bpermanent\b', re.I),         +1.8),
    (re.compile(r'\bwipe\b', re.I),              +2.3),
    (re.compile(r'\berase\b', re.I),             +1.9),
    (re.compile(r'\bskip.?tests?\b', re.I),      +2.6),
    (re.compile(r'\bforce.?deploy\b', re.I),     +2.2),
    (re.compile(r'\bdisable\b', re.I),           +1.8),
    (re.compile(r'\bbypass\b', re.I),            +2.0),
    (re.compile(r'\boverride\b', re.I),          +1.6),
    (re.compile(r'\bno.?ticket\b', re.I),        +2.1),
    (re.compile(r'\bnot provided\b', re.I),      +2.0),
    (re.compile(r'\bgrant.{0,30}admin\b', re.I), +2.3),
    (re.compile(r'\bgrant.{0,30}super\b', re.I), +2.5),
    (re.compile(r'\brate.?limit.{0,10}0\b', re.I), +2.8),
    (re.compile(r'\bfull.?rollout\b', re.I),     +1.4),
    (re.compile(r'\bproduction\b', re.I),        +1.2),
    (re.compile(r'\bprod\b', re.I),              +1.1),
    (re.compile(r'\bcredential\b', re.I),        +1.5),
    (re.compile(r'\bpassword\b', re.I),          +1.7),
    (re.compile(r'\bsecret\b', re.I),            +1.4),
    (re.compile(r'\bapi.?key\b', re.I),          +1.6),
    # Medium-risk
    (re.compile(r'\bupdate\b', re.I),            +0.8),
    (re.compile(r'\bdeploy\b', re.I),            +0.6),
    (re.compile(r'\bgrant\b', re.I),             +0.9),
    (re.compile(r'\bpermission\b', re.I),        +0.5),
    (re.compile(r'\baccess\b', re.I),            +0.4),
    (re.compile(r'\bconfig\b', re.I),            +0.3),
    # Injection markers
    (re.compile(r'ignore.{0,20}instructions?', re.I), +3.0),
    (re.compile(r'you are now', re.I),           +2.5),
    (re.compile(r'jailbreak', re.I),             +3.0),
    # Risk reducers
    (re.compile(r'\bselect\b', re.I),            -1.5),
    (re.compile(r'\bread.?only\b', re.I),        -2.0),
    (re.compile(r'\bdry.?run\b', re.I),          -1.8),
    (re.compile(r'\bcanary\b', re.I),            -0.8),
    (re.compile(r'\bblue.?green\b', re.I),       -0.6),
    (re.compile(r'\bstaging\b', re.I),           -0.9),
    (re.compile(r'\bdev\b', re.I),               -0.7),
    (re.compile(r'\bbackup\b', re.I),            -1.0),
    (re.compile(r'\bconfirm(ation)?\b', re.I),   -1.2),
    (re.compile(r'\bapproval\b', re.I),          -1.1),
    (re.compile(r'\bticket\b', re.I),            -1.3),
    (re.compile(r'\bjira.?\d+\b', re.I),         -1.5),
    (re.compile(r'\breadonly\b', re.I),          -1.8),
    (re.compile(r'\bexplain\b', re.I),           -1.2),
    (re.compile(r'\bdescribe\b', re.I),          -1.0),
    (re.compile(r'\brollback\b', re.I),          -1.5),  # rollback = recovery = safe
]


# ── Structured feature extraction ─────────────────────────────────────────────

def _extract_structured_features(action: str, context: str) -> dict[str, float]:
    """
    Extract 20 named binary/numeric features from action + context.
    These complement the term weights and capture relational patterns
    (e.g. destructive operation AND production target).
    """
    a = action.lower()
    c = context.lower()
    combined = f"{a} {c}"

    is_prod   = int(bool(re.search(r'\b(production|prod\b|live\b)', combined)))
    is_dev    = int(bool(re.search(r'\b(dev|staging|sandbox|test\b)', combined)))
    has_ticket = int(bool(re.search(r'[A-Z]{2,8}-\d{2,8}', action, re.I)))
    has_confirmation = int(bool(re.search(r'\b(confirm|approval|explicit)\b', combined)))

    destructive_op = int(bool(re.search(
        r'\b(delete|drop|truncate|wipe|purge|erase)\b', a
    )))
    write_op = int(bool(re.search(r'\b(update|insert|modify|change|set)\b', a)))
    read_op  = int(bool(re.search(r'^(select|show|describe|explain|fetch|get|list)', a.strip())))
    deploy_op = int(bool(re.search(r'\bdeploy\b', a)))
    perm_op   = int(bool(re.search(r'\b(grant|revoke|assign)\b', a)))
    config_op = int(bool(re.search(r'\b(update|set|change).{0,30}config\b', a)))

    skip_safety = int(bool(re.search(r'\b(skip|bypass|force|override|ignore)\b', a)))
    no_ticket_text = int(bool(re.search(r'not provided|no ticket|without ticket', combined)))

    return {
        # Risk compounders — high weight when combined
        "destructive_x_prod":    float(destructive_op and is_prod),        # +3.0
        "perm_grant_no_ticket":  float(perm_op and not has_ticket),         # +2.5
        "deploy_skip_safety":    float(deploy_op and skip_safety),          # +3.0
        "config_x_prod":         float(config_op and is_prod),              # +1.8
        "write_x_prod":          float(write_op and is_prod),               # +1.2
        # Risk mitigators — reduce score when present
        "has_ticket":            float(has_ticket),                         # -1.5
        "has_confirmation":      float(has_confirmation),                   # -1.2
        "is_read_only":          float(read_op),                            # -2.0
        "is_dev_env":            float(is_dev),                             # -0.8
        # Raw indicators
        "is_prod":               float(is_prod),                            # +1.0
        "is_destructive":        float(destructive_op),                     # +1.5
        "is_perm_change":        float(perm_op),                            # +0.8
        "no_ticket_text":        float(no_ticket_text),                     # +2.0
        "skip_safety":           float(skip_safety),                        # +1.5
    }


# Weights for structured features (matching _extract_structured_features keys)
_STRUCTURED_WEIGHTS: dict[str, float] = {
    "destructive_x_prod":    3.0,
    "perm_grant_no_ticket":  2.5,
    "deploy_skip_safety":    3.0,
    "config_x_prod":         1.8,
    "write_x_prod":          1.2,
    "has_ticket":           -1.5,
    "has_confirmation":     -1.2,
    "is_read_only":         -2.0,
    "is_dev_env":           -0.8,
    "is_prod":               1.0,
    "is_destructive":        1.5,
    "is_perm_change":        0.8,
    "no_ticket_text":        2.0,
    "skip_safety":           1.5,
}

_BIAS = -3.5   # Tuned so neutral actions score ~0.03 (safe by default)

# Thresholds for verdict decision
BLOCK_THRESHOLD   = 0.72
ALLOW_THRESHOLD   = 0.28


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# ── Public API ─────────────────────────────────────────────────────────────────

def score(
    action: str,
    context: str = "",
    operator_constraints: Optional[list[str]] = None,
) -> MLScore:
    """
    Score an agent action for safety risk.

    Returns:
        MLScore with risk_score in [0, 1] and a verdict.
    """
    combined = f"{action} {context}"

    # ── Term-weight pass ──
    term_scores: list[tuple[str, float]] = []
    logit = _BIAS
    for pattern, weight in _TERM_WEIGHTS:
        if pattern.search(combined):
            term_scores.append((pattern.pattern, weight))
            logit += weight

    # ── Structured feature pass ──
    features = _extract_structured_features(action, context)
    feature_scores: list[tuple[str, float]] = []
    for fname, fval in features.items():
        w = _STRUCTURED_WEIGHTS.get(fname, 0.0)
        contribution = w * fval
        if contribution != 0.0:
            feature_scores.append((fname, contribution))
        logit += contribution

    # Optional: boost if operator constraints are provided but clearly violated
    if operator_constraints:
        a_lower = action.lower()
        for constraint in operator_constraints:
            c_lower = constraint.lower()
            if "without" in c_lower and "confirmation" in c_lower:
                if any(k in a_lower for k in ("delete", "drop", "truncate")):
                    logit += 1.5
            if "ticket" in c_lower and "grant" in a_lower:
                if not re.search(r'[A-Z]{2,8}-\d{2,8}', action, re.I):
                    logit += 1.5

    risk_score = _sigmoid(logit)

    # Build top-5 contributing features for explainability
    all_contributions = term_scores + feature_scores
    all_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = all_contributions[:5]

    if risk_score >= BLOCK_THRESHOLD:
        verdict = "block"
    elif risk_score <= ALLOW_THRESHOLD:
        verdict = "allow"
    else:
        verdict = "uncertain"

    return MLScore(
        risk_score=risk_score,
        verdict=verdict,
        top_features=top_features,
    )
