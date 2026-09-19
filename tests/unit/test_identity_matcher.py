"""Phase 2: identity matcher unit tests (no DB).

The false-merge regression test here targets the failure mode the problem
statement names explicitly: two DISTINCT people must never merge because they
share a surname.
"""
from __future__ import annotations

from datetime import date

from src.identity.matcher import IdentityRecord, match


def test_exact_email_merges_case_insensitive() -> None:
    a = IdentityRecord("C1", "core_banking", full_name="Marcus Vance", email="Marcus@X.com")
    b = IdentityRecord("C2", "support_logs", full_name="marcus vance", email="  marcus@x.com ")
    r = match(a, b)
    assert r.matched and r.method.value == "exact_email" and r.confidence >= 0.95


def test_exact_phone_merges_ignoring_formatting() -> None:
    a = IdentityRecord("C1", "core_banking", phone="+1 (555) 123-4567")
    b = IdentityRecord("C2", "support_logs", phone="5551234567")
    r = match(a, b)
    assert r.matched and r.method.value == "exact_phone"


def test_false_merge_regression_shared_surname_different_people() -> None:
    """Same surname, different DOB/email/phone => must NOT merge."""
    a = IdentityRecord(
        "C1", "core_banking",
        full_name="David Chen", email="david.chen@personal.com",
        phone="555-0101", dob=date(1985, 3, 10),
    )
    b = IdentityRecord(
        "C2", "support_logs",
        full_name="Deborah Chen", email="deb.chen@other.org",
        phone="555-0909", dob=date(1979, 11, 2),
    )
    r = match(a, b)
    assert r.matched is False, "false merge: distinct people sharing a surname merged"
    assert r.score < 0 or not r.matched


def test_probabilistic_merge_same_person_varied_records() -> None:
    """Same person, slightly different name spelling, same DOB/email => merge."""
    a = IdentityRecord("C1", "core_banking", full_name="Marcus Vance", email="mv@x.com", dob=date(1978, 6, 15))
    b = IdentityRecord("C2", "support_logs", full_name="Marcus Vannce", email="mv@x.com", dob=date(1978, 6, 15))
    r = match(a, b)
    assert r.matched and r.confidence >= 0.9


def test_no_comparable_fields_never_merges() -> None:
    a = IdentityRecord("C1", "core_banking", full_name="A Name")
    b = IdentityRecord("C2", "support_logs", full_name="B Other")
    r = match(a, b)
    assert r.matched is False


def test_conflicting_deterministic_keys_prevent_merge() -> None:
    """Different emails + different phones overpower a similar name."""
    a = IdentityRecord("C1", "core_banking", full_name="John Smith", email="js1@x.com", phone="555-0001")
    b = IdentityRecord("C2", "support_logs", full_name="John Smyth", email="js2@x.com", phone="555-0002")
    r = match(a, b)
    assert r.matched is False
