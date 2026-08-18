"""Duplicate transaction detection.

Brief-specified key: date + description + amount + reference (Transaction.dedup_key,
defined in Phase 2 alongside the schema it belongs to — colocating the key
definition with the class it describes avoids two modules silently disagreeing about
what "duplicate" means).

Note on what this deliberately does NOT do: it doesn't try to detect *near*-duplicates
(same amount, description off by one character, date one day apart) — that's a fuzzy-
matching problem with real false-positive risk (two identical $12.50 coffee purchases
on consecutive days are not a duplicate), and fuzzy duplicate detection is exactly the
kind of judgment call better suited to the CrewAI Anomaly Analyst agent (Phase 9)
interpreting this module's exact-match output alongside broader context, rather than
this module trying to encode fuzzy heuristics itself.
"""

from __future__ import annotations

from collections import defaultdict

from app.schemas.anomaly import Anomaly
from app.schemas.transaction import Transaction


def find_duplicate_transactions(transactions: list[Transaction]) -> list[Anomaly]:
    """Group transactions by exact dedup_key (date, description, debit, credit,
    reference) and flag any group with more than one member.

    Returns one Anomaly per duplicate GROUP (not per duplicate transaction) — a group
    of 3 identical rows produces 1 anomaly listing all 3 indices, not 3 separate
    anomalies, since a human reviewer wants to see "these 3 rows look identical" as
    one fact, not three redundant alerts about the same underlying issue.
    """
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, txn in enumerate(transactions):
        groups[txn.dedup_key].append(index)

    anomalies: list[Anomaly] = []
    for key, indices in groups.items():
        if len(indices) > 1:
            date, description, debit, credit, reference = key
            description_summary = f"{date} | {description} | debit={debit} credit={credit}"
            anomalies.append(Anomaly.duplicate(indices, description_summary))

    return anomalies
