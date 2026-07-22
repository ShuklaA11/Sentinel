"""Local watcher: keep a browser open, fill newly-queued Greenhouse applications into
open tabs, and email you when they're ready to submit.

This is the "I don't run it manually" surface. Start it once on your Mac and leave it
running: each cycle it (optionally) polls for new roles, fills every new high-fit
Greenhouse listing into its own tab, leaves that tab open on the completed form, and
emails you. You come back, review the waiting tabs, solve any captcha, and click Submit.
Nothing submits on its own.

Hard constraint: a filled form only lives inside this open browser. If the machine sleeps
or you quit the window, the fills are lost and re-created next cycle. It must run on your
machine with a real display (not a headless server) — that's where you click Submit.

Run:
  python -m src.apply_watcher --poll --interval 900     # poll + fill + notify, every 15 min
  python -m src.apply_watcher --once                    # one cycle over the current queue
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time

from src import apply_driver, apply_queue, notify

log = logging.getLogger("apply_watcher")


def _poll_once() -> None:
    """Refresh the queue by running the detector (detect -> rank -> enqueue)."""
    subprocess.run([sys.executable, "-m", "src.run"], check=False)


def _prepare(page, row: dict, profile: dict, voice: str, resume_path: str) -> list[dict]:
    """Fill one queued row's form into `page` (left open). Returns per-field decisions."""
    page.goto(row["url"], timeout=30000)
    apply_driver._reach_form(page)
    return [apply_driver._apply_field(page, f, profile, voice, resume_path)
            for f in apply_driver.extract_fields(page)]


def watch(interval: int = 900, poll: bool = False, once: bool = False) -> None:
    from playwright.sync_api import sync_playwright  # lazy import: package is optional

    profile, voice = apply_driver._load_profile(), apply_driver._load_voice()
    resume_path = (profile.get("facts") or {}).get("resume_path", "")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)   # must be visible: you click Submit
        ctx = browser.new_context()
        while True:
            if poll:
                _poll_once()
            rows = apply_queue.load_queue()
            todo = [r for r in rows if r.get("status") == "queued" and r.get("ats") == "greenhouse"]
            prepared: list[dict] = []
            for row in todo:
                page = ctx.new_page()   # left open for your review + submit
                try:
                    decisions = _prepare(page, row, profile, voice, resume_path)
                    shot = apply_driver._shot(page, row["id"])
                    rows = apply_queue.update_row(rows, row["id"], status="prepared",
                                                  screenshot=shot, note=apply_driver._note(decisions),
                                                  attempts=(row.get("attempts") or 0) + 1)
                    prepared.append(row)
                    log.info("prepared %s (tab left open)", row["company"])
                except Exception as exc:  # noqa: BLE001 — one bad row must not stop the batch
                    rows = apply_queue.update_row(rows, row["id"], status="failed", note=f"{exc}"[:200])
                    page.close()
                    log.error("failed %s: %s", row.get("company"), exc)
                finally:
                    apply_queue.save_queue(rows)
            if prepared:
                notify.send_apply_ready(prepared)
                log.info("%d ready — emailed you; review the open tabs and click Submit", len(prepared))
            if once:
                break
            time.sleep(interval)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Local apply watcher: fill + notify, you submit")
    ap.add_argument("--interval", type=int, default=900, help="seconds between cycles (default 900)")
    ap.add_argument("--poll", action="store_true", help="run the detector each cycle to refresh the queue")
    ap.add_argument("--once", action="store_true", help="run one cycle over the current queue and exit")
    args = ap.parse_args()
    watch(interval=args.interval, poll=args.poll, once=args.once)


if __name__ == "__main__":
    main()
