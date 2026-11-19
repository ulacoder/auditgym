import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditgym.config import ERROR_TYPES  # noqa: E402
from auditgym.generator import generate  # noqa: E402

ALL = {"errors": {t: 1 for t in ERROR_TYPES},
       "interactions": {"fx_invoice_with_partial_payment": 1, "partial_fx_payment_at_boundary": 1}}
CLEAN = {}


@pytest.fixture(scope="session")
def ep_all():
    return generate(4242, ALL)


@pytest.fixture(scope="session")
def ep_clean():
    return generate(4243, CLEAN)


def oracle_findings(ep, ev_index=0):
    return [{"type": t["type"], "affected_ids": list(t["affected_ids"]),
             "claimed_amount_minor": t["expected_amount_minor"], "currency": t["currency"],
             "evidence_ids": list(t["evidence_sets"][min(ev_index, len(t["evidence_sets"]) - 1)]), "explanation": "x"} for t in ep.truth["issues"]]


def sub(ep, findings):
    return {"episode_id": ep.episode_id, "findings": findings}
