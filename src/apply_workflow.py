"""One-shot apply session: fill what we can, open the rest, you submit.

Run this once on your Mac when you want to knock out today's high-fit roles:

  python -m src.apply_workflow

It polls, opens a visible browser, and in that one window:
  * FILLS every high-fit Greenhouse/Lever/Ashby role into its own tab (resume + screening),
  * OPENS the high-fit roles the driver can't fill (repo-sourced / other ATS) as plain tabs
    for you to apply to by hand,
  * emails you a summary, and holds the browser open until you're done.

Nothing submits on its own. A human (you) reviews every tab, solves any captcha, and clicks
Submit. Must run on your machine with a display — the filled tabs only live in this window.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys

from src import apply_driver, apply_queue, notify

log = logging.getLogger("apply_workflow")

LISTINGS = os.path.join(apply_driver.ROOT, "data", "listings.csv")
DEFAULT_THRESHOLD = 85
DEFAULT_MANUAL_CAP = 15   # don't flood the browser with dozens of manual tabs


def high_fit_manual(threshold: int = DEFAULT_THRESHOLD, cap: int = DEFAULT_MANUAL_CAP) -> list[dict]:
    """High-fit listings the driver can't fill (unsupported ATS / repo-sourced), top `cap`."""
    if not os.path.exists(LISTINGS):
        return []
    rows = list(csv.DictReader(open(LISTINGS)))

    def hi(r: dict) -> bool:
        s = (r.get("score") or "").strip()
        return s.isdigit() and int(s) >= threshold

    manual = [r for r in rows if hi(r)
              and apply_queue.ats_from(r.get("source", "")) not in apply_queue.SUPPORTED_ATS]
    manual.sort(key=lambda r: -int(r["score"]))
    return manual[:cap]


def run_session(poll: bool = True, threshold: int = DEFAULT_THRESHOLD,
                manual_cap: int = DEFAULT_MANUAL_CAP) -> None:
    from playwright.sync_api import sync_playwright  # lazy import: package is optional

    if poll:
        log.info("polling for fresh listings...")
        subprocess.run([sys.executable, "-m", "src.run"], check=False)

    profile, voice = apply_driver._load_profile(), apply_driver._load_voice()
    resume_path = (profile.get("facts") or {}).get("resume_path", "")
    rows = apply_queue.load_queue()
    fillable = [r for r in rows if r.get("status") == "queued"
                and r.get("ats") in apply_queue.SUPPORTED_ATS]
    manual = high_fit_manual(threshold, manual_cap)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)   # visible: you submit in these tabs
        ctx = browser.new_context()

        filled: list[dict] = []
        for row in fillable:
            page = ctx.new_page()                      # left open for review + submit
            try:
                page.goto(row["url"], timeout=30000)
                apply_driver._reach_form(page)
                decisions = [apply_driver._apply_field(page, f, profile, voice, resume_path)
                             for f in apply_driver.extract_fields(page)]
                apply_driver._shot(page, row["id"])
                rows = apply_queue.update_row(rows, row["id"], status="prepared",
                                              note=apply_driver._note(decisions),
                                              attempts=(row.get("attempts") or 0) + 1)
                filled.append(row)
                log.info("filled %s (tab open)", row["company"])
            except Exception as exc:  # noqa: BLE001 — one bad row must not stop the session
                rows = apply_queue.update_row(rows, row["id"], status="failed", note=f"{exc}"[:200])
                page.close()
                log.error("failed %s: %s", row.get("company"), exc)
            finally:
                apply_queue.save_queue(rows)

        opened: list[dict] = []
        for r in manual:
            page = ctx.new_page()                      # just opened, for manual apply
            try:
                page.goto(r["url"], timeout=30000)
                opened.append(r)
            except Exception as exc:  # noqa: BLE001 — a dead link must not stop the session
                page.close()
                log.warning("couldn't open %s: %s", r.get("company"), exc)

        if filled:
            notify.send_apply_ready(filled)
        print(f"\n{len(filled)} filled (ready to submit) + {len(opened)} opened for manual apply.")
        print("Review the tabs, solve any captcha, and click Submit yourself.")
        input("Press Enter here when you're done to close the browser...")
        browser.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="One-shot apply session: fill + open, you submit")
    ap.add_argument("--no-poll", action="store_true", help="skip the detector, use the current queue")
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="min fit score")
    ap.add_argument("--manual-cap", type=int, default=DEFAULT_MANUAL_CAP,
                    help="max manual tabs to open (avoid flooding the browser)")
    args = ap.parse_args()
    run_session(poll=not args.no_poll, threshold=args.threshold, manual_cap=args.manual_cap)


if __name__ == "__main__":
    main()
