"""Mark stale tracker listings closed by comparing against each source's current poll.

Signal: a previously-open listing whose ATS source polled *healthily* this run but no
longer returns its id is very likely closed. The guard against false-closes is that a
source returning zero/errored results — indistinguishable, since fetchers swallow errors
and return [] — is treated as *ambiguous* and can never close its listings. Repo
aggregators bundle many companies under churny derived ids, so they are excluded from
presence-based liveness (deferred to an optional URL re-check we don't ship in v1).

All functions here are pure (no network, no I/O) so the classification is unit-testable.
"""
from __future__ import annotations

# Providers whose per-slug poll returns the full current opening set, so a missing id is
# a trustworthy "closed" signal. Repo aggregators (source "repo:*") are intentionally out.
ATS_SOURCES = frozenset(
    {"greenhouse", "lever", "ashby", "smartrecruiters", "workable", "bamboohr", "workday"}
)


def poll_unit(source: str, company: str) -> str:
    """Reconstruct the fetch_all() task label a listing belongs to (e.g. 'greenhouse:ramp')."""
    return f"{source}:{company}"


def healthy_units(stats: dict) -> set[str]:
    """Source labels that returned at least one listing this poll.

    count==0 is ambiguous (healthy-but-empty vs. errored), so only count>0 units are
    trusted to close their missing listings — this is the outage guard.
    """
    return {label for label, count in stats.items() if count > 0}


def present_ids(raw: list[dict]) -> set[str]:
    """Ids the sources currently return, taken from RAW (pre-filter) fetch output.

    Must be pre-filter: a still-live listing that merely stopped matching filters this run
    would otherwise look absent and be falsely closed.
    """
    return {l["id"] for l in raw}


def classify(rows: list[dict], present: set[str], healthy: set[str]) -> tuple[set[str], set[str]]:
    """Split tracker rows into (closed_ids, reopened_ids) for direct-ATS units only.

    - close:  status != closed, unit healthy, id absent from `present`      -> close
    - reopen: status == closed, unit healthy, id present again (self-heal)   -> reopen
    Rows on repo/unknown sources, or on units that weren't healthy this poll, are left
    untouched (repo-exclusion + outage/config-drift guard).
    """
    closed_ids: set[str] = set()
    reopened_ids: set[str] = set()
    for r in rows:
        source = r.get("source", "")
        if source not in ATS_SOURCES:
            continue  # repo / unknown source: no presence-based signal
        if poll_unit(source, r.get("company", "")) not in healthy:
            continue  # outage or not-polled this run: ambiguous, never touch
        rid = r.get("id", "")
        is_closed = r.get("status") == "closed"
        if rid in present:
            if is_closed:
                reopened_ids.add(rid)
        elif not is_closed:
            closed_ids.add(rid)
    return closed_ids, reopened_ids
