"""Identity matching: deterministic-first, probabilistic fallback (Phase 2).

Per RQ-I1 (research finding) and the PS's named cross-contamination failure
mode: deterministic exact-match on unique identifiers first (low false-positive
rate), then probabilistic scoring (Fellegi-Sunter log-likelihood-ratio with
Jaro-Winkler string similarity) only for genuinely ambiguous cases - always
producing a stored, auditable confidence. Merges below threshold are refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from rapidfuzz.distance import JaroWinkler


class MatchMethod(str, Enum):
    FIRST_SEEN = "first_seen"
    EXACT_EMAIL = "exact_email"
    EXACT_PHONE = "exact_phone"
    EXACT_KYC_DOC = "exact_kyc_doc"
    PROBABILISTIC = "probabilistic"


@dataclass(frozen=True)
class IdentityRecord:
    """Attributes of one identity record (source-system level)."""

    source_customer_id: str
    source_system: str
    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    dob: date | None = None
    kyc_doc_number: str | None = None

    @property
    def canonical_email(self) -> str | None:
        return self.email.strip().lower() if self.email else None

    @property
    def canonical_phone(self) -> str | None:
        if not self.phone:
            return None
        digits = "".join(ch for ch in self.phone if ch.isdigit())
        if len(digits) == 11 and digits.startswith("1"):  # US country code
            digits = digits[1:]
        return digits or None


@dataclass(frozen=True)
class MatchResult:
    """Outcome of comparing two records - always auditable."""

    matched: bool
    method: MatchMethod
    confidence: float
    score: float
    reason: str


# Fellegi-Sunter parameters: m = P(agree | match), u = P(agree | non-match).
# Log-likelihood-ratio per field; deterministic keys short-circuit.
FIELD_WEIGHTS = {
    "email": (0.97, 0.001),
    "phone": (0.95, 0.005),
    "name": (0.90, 0.01),
    "dob": (0.95, 0.01),
    "kyc_doc": (0.99, 0.0005),
}
NAME_SIMILARITY_THRESHOLD = 0.90
MERGE_THRESHOLD = 2.0  # LLR threshold; refusing merges below this prevents false merges


def _llr(agree: bool, field: str) -> float:
    import math

    m, u = FIELD_WEIGHTS[field]
    return math.log(m / u) if agree else math.log((1 - m) / (1 - u))


def deterministic_match(a: IdentityRecord, b: IdentityRecord) -> MatchResult | None:
    """Exact-match on unique identifiers: email, phone, KYC document number.

    Returns None when no deterministic key is comparable - the caller then
    falls back to probabilistic scoring. Deterministic matches are high
    confidence by construction (m/u ratios of 200-2000x).
    """
    if a.canonical_email and a.canonical_email == b.canonical_email:
        return MatchResult(True, MatchMethod.EXACT_EMAIL, 0.99, _llr(True, "email"), "exact email match")
    if a.canonical_phone and a.canonical_phone == b.canonical_phone:
        return MatchResult(True, MatchMethod.EXACT_PHONE, 0.98, _llr(True, "phone"), "exact phone match")
    if a.kyc_doc_number and a.kyc_doc_number == b.kyc_doc_number:
        return MatchResult(True, MatchMethod.EXACT_KYC_DOC, 0.995, _llr(True, "kyc_doc"), "exact kyc document match")
    return None


def probabilistic_match(a: IdentityRecord, b: IdentityRecord) -> MatchResult:
    """Fellegi-Sunter LLR over comparable fields; merge only above threshold.

    Contradicting deterministic keys (different emails, different DOBs) count
    as strong *disagreement* evidence and drive the score below threshold -
    this is what prevents false merges of distinct people who share a surname.
    """
    score = 0.0
    compared = 0
    reasons: list[str] = []

    if a.canonical_email and b.canonical_email:
        agree = a.canonical_email == b.canonical_email
        score += _llr(agree, "email")
        compared += 1
        reasons.append(f"email:{'agree' if agree else 'disagree'}")
    if a.canonical_phone and b.canonical_phone:
        agree = a.canonical_phone == b.canonical_phone
        score += _llr(agree, "phone")
        compared += 1
        reasons.append(f"phone:{'agree' if agree else 'disagree'}")
    if a.kyc_doc_number and b.kyc_doc_number:
        agree = a.kyc_doc_number == b.kyc_doc_number
        score += _llr(agree, "kyc_doc")
        compared += 1
        reasons.append(f"kyc_doc:{'agree' if agree else 'disagree'}")
    if a.dob and b.dob:
        agree = a.dob == b.dob
        score += _llr(agree, "dob")
        compared += 1
        reasons.append(f"dob:{'agree' if agree else 'disagree'}")
    if a.full_name and b.full_name:
        sim = JaroWinkler.similarity(a.full_name.strip().lower(), b.full_name.strip().lower())
        agree = sim >= NAME_SIMILARITY_THRESHOLD
        score += _llr(agree, "name")
        compared += 1
        reasons.append(f"name:jw={sim:.3f}")

    matched = compared > 0 and score >= MERGE_THRESHOLD
    # map LLR to a bounded confidence in (0,1): sigmoid, floor at 0.5 when matched
    import math

    confidence = 1 / (1 + math.exp(-score)) if matched else max(0.0, min(0.5, 1 / (1 + math.exp(-score))))
    return MatchResult(
        matched=matched,
        method=MatchMethod.PROBABILISTIC,
        confidence=round(confidence, 4),
        score=round(score, 4),
        reason=";".join(reasons) or "no comparable fields",
    )


def match(a: IdentityRecord, b: IdentityRecord) -> MatchResult:
    """Deterministic-first, probabilistic fallback (RQ-I1 pipeline order)."""
    det = deterministic_match(a, b)
    if det is not None:
        return det
    return probabilistic_match(a, b)
