"""Flush the pending-digest outbox: email the queued (sub-threshold) listings, then
clear the queue. Run on a schedule a few times a day. Run: python -m src.digest

The queue is cleared ONLY after the email is confirmed sent — so a failed/skipped
send leaves the listings queued for the next run instead of dropping them.
"""
from __future__ import annotations

import logging

from . import notify, store

log = logging.getLogger("digest")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    pending = store.load_pending()
    if not pending:
        print("no pending listings — nothing to digest")
        return

    if notify.send_digest(pending):
        store.clear_pending()
        print(f"digested {len(pending)} listings, cleared the queue")
    else:
        print(f"{len(pending)} pending but email was skipped/failed — keeping the queue")


if __name__ == "__main__":
    main()
