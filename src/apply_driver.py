"""Playwright execution driver for the auto-apply pipeline (Greenhouse adapter).

This is the browser substrate for autonomous runs. Unlike the Claude-in-Chrome shadow
playbook (which needs an agent driving each click), this is a plain script: it reads the
apply queue, drives a real browser, and reuses all existing logic (apply_fill.plan_field,
essays.draft_answer, apply_queue). Crucially, resume attach works here via Playwright's
set_input_files, which the Claude-in-Chrome file_upload tool cannot do.

Shadow by default: fills every field it confidently can, attaches the resume, screenshots
the completed form, and STOPS before submit. The live-submit flip plus captcha/login
guards land in a later step (4c); --submit is accepted now but intentionally inert.

Run:
  python -m src.apply_driver --limit 1            # headless shadow: fill, screenshot, stop
  python -m src.apply_driver --review             # headful: fill, then pause for you to submit

--review is the assisted endpoint: it opens a visible window, fills each queued form
(resume included), and waits on the completed form so YOU review, solve any captcha, and
click Submit yourself. That sidesteps captcha walls entirely, since a human is present.
"""
from __future__ import annotations

import argparse
import logging
import os
import re

import yaml

from src import apply_fill, apply_queue, essays

log = logging.getLogger("apply_driver")

ROOT = os.path.dirname(os.path.dirname(__file__))
PROFILE_PATH = os.path.join(ROOT, "profile", "profile.yml")
VOICE_PATH = os.path.join(ROOT, "profile", "voice.md")
SHOTS_DIR = os.path.join(ROOT, "data", "apply_shots")

# One JS pass per input: resolve its label, type, required flag, and (for selects)
# its option texts. Runs in the page so label resolution sees the real DOM.
_FIELD_JS = r"""
el => {
  const tag = el.tagName.toLowerCase();
  const type = (el.getAttribute('type') || (tag === 'select' ? 'select' : tag)).toLowerCase();
  if (type === 'hidden') return null;
  if (el.closest('.select-shell')) return null;   // react-select internals handled separately
  let label = '';
  if (el.id) { const l = document.querySelector('label[for="' + CSS.escape(el.id) + '"]'); if (l) label = l.innerText; }
  if (!label && el.getAttribute('aria-label')) label = el.getAttribute('aria-label');
  if (!label) { const l = el.closest('label'); if (l) label = l.innerText; }
  if (!label) { const c = el.closest('div,li,fieldset'); if (c) { const l = c.querySelector('label,legend'); if (l) label = l.innerText; } }
  const required = el.hasAttribute('required') || el.getAttribute('aria-required') === 'true';
  let options = null;
  if (tag === 'select') options = Array.from(el.options).map(o => o.textContent.trim()).filter(Boolean);
  return { tag, type, label: (label || '').replace(/\*/g, ' ').replace(/\s+/g, ' ').trim(), required, options };
}
"""


def _load_profile() -> dict:
    with open(PROFILE_PATH) as f:
        return yaml.safe_load(f) or {}


def _load_voice() -> str:
    return open(VOICE_PATH).read() if os.path.exists(VOICE_PATH) else ""


# react-select (Greenhouse) has no <label for>; walk up to the nearest label in the
# enclosing question block.
_SELECT_LABEL_JS = r"""el => {
  let n = el;
  for (let i = 0; i < 6 && n; i++) { n = n.parentElement; if (!n) break;
    const l = n.querySelector('label'); if (l && l.innerText.trim()) return l.innerText; }
  return '';
}"""


def extract_fields(page) -> list[dict]:
    """Enumerate the application form's fillable fields, each with a live handle.

    Handles two shapes: react-select widgets (Greenhouse's `.select-shell`, filled by
    open-menu-then-click) and standard input/textarea/select. Public for testing against
    a local HTML fixture (no network needed).
    """
    out: list[dict] = []
    for shell in page.query_selector_all(".select-shell"):
        ctrl = shell.query_selector(".select__control")
        if not ctrl:
            continue
        label = re.sub(r"\s+", " ", shell.evaluate(_SELECT_LABEL_JS) or "").replace("*", "").strip()
        out.append({"tag": "select-widget", "type": "select-widget", "label": label,
                    "required": False, "options": None, "_handle": ctrl})
    for h in page.query_selector_all("form input, form textarea, form select"):
        info = h.evaluate(_FIELD_JS)
        if not info:
            continue
        info["_handle"] = h
        out.append(info)
    return out


def _fill_select_widget(page, f: dict, profile: dict) -> dict:
    """Open a react-select, read its real options, resolve, and click the match."""
    ctrl = f["_handle"]
    try:
        ctrl.click()
        page.wait_for_timeout(350)
    except Exception as exc:  # noqa: BLE001
        return {"label": f["label"], "action": "needs_human", "reason": f"open failed: {exc}"[:80]}
    opt_els = page.query_selector_all(".select__option")
    options = [o.inner_text().strip() for o in opt_els]
    fd = {"label": f["label"], "type": "select", "required": f["required"], "options": options}
    d = apply_fill.plan_field(fd, profile)
    if d["action"] == "fill":
        for o in opt_els:
            if o.inner_text().strip() == d["value"]:
                o.click()
                page.wait_for_timeout(150)
                return d
        d = {**d, "action": "needs_human", "reason": "resolved option not in menu"}
    try:
        ctrl.press("Escape")   # close the menu we opened
    except Exception:  # noqa: BLE001
        pass
    return d


def _reach_form(page) -> None:
    """Best-effort: click an Apply button if the form isn't already shown, then wait."""
    for sel in ("button:has-text('Apply for this job')", "a:has-text('Apply for this job')",
                "button:has-text('Apply Now')"):
        btn = page.query_selector(sel)
        if btn:
            try:
                btn.click()
                page.wait_for_timeout(1500)
                break
            except Exception:  # noqa: BLE001 — apply button is optional
                pass
    page.wait_for_selector("form input, form textarea", timeout=15000)


def _apply_field(page, f: dict, profile: dict, voice: str, resume_path: str) -> dict:
    """Decide + perform the action for one field. Returns the decision (for the note)."""
    if f["type"] == "select-widget":
        return _fill_select_widget(page, f, profile)
    fd = {"label": f["label"], "type": f["type"], "required": f["required"], "options": f.get("options")}
    d = apply_fill.plan_field(fd, profile)
    h = f["_handle"]
    try:
        if d["action"] == "fill":
            if f["type"] == "file":
                if resume_path and os.path.exists(resume_path):
                    h.set_input_files(resume_path)          # <-- resume attach, natively
                else:
                    return {**d, "action": "needs_human", "reason": "resume file missing"}
            elif f["tag"] == "select":
                try:
                    h.select_option(label=d["value"])
                except Exception:  # noqa: BLE001 — fall back to matching by value
                    h.select_option(d["value"])
            else:
                h.fill(str(d["value"]))
        elif d["action"] == "draft":
            ans = essays.draft_answer(d["question"], profile, voice)
            if ans:
                h.fill(ans)
                d = {**d, "drafted": True}
            else:
                d = {**d, "action": "needs_human", "reason": "essay draft unavailable"}
    except Exception as exc:  # noqa: BLE001 — one bad field must not abort the row
        d = {**d, "action": "needs_human", "reason": f"fill error: {exc}"[:120]}
    return d


def _note(decisions: list[dict]) -> str:
    filled = [d["label"] for d in decisions if d["action"] == "fill"]
    drafted = [d["label"] for d in decisions if d.get("drafted")]
    human = [d["label"] for d in decisions if d["action"] == "needs_human"]
    parts = [f"filled={len(filled)}"]
    if drafted:
        parts.append("drafted=" + ",".join(drafted))
    if human:
        parts.append("needs_human=" + ",".join(human))
    return "SHADOW: " + "; ".join(parts) + "; NOT submitted"


def _hold_for_review(row: dict) -> str:
    """Pause on a filled form so the human reviews, solves any captcha, and submits.

    Only called in --review mode. Returns the status to record. Blocks on input(),
    so the browser window stays open and interactive until the human responds.
    """
    print(f"\n>>> {row.get('company')} — {row.get('title')}")
    print("    Form is filled in the browser. Review it, add anything left blank,")
    print("    solve any captcha, and click Submit yourself.")
    ans = input("    [Enter]=reviewed   s=submitted   q=quit: ").strip().lower()
    if ans == "q":
        return "__quit__"
    return "submitted" if ans == "s" else "reviewed"


def _shot(page, id_: str) -> str:
    os.makedirs(SHOTS_DIR, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", id_)
    path = os.path.join(SHOTS_DIR, f"{safe}.png")
    page.screenshot(path=path, full_page=True)
    return path


def run(limit: int | None = None, headful: bool = False, submit: bool = False,
        review: bool = False) -> None:
    if submit:
        log.warning("--submit is not enabled yet (shadow only); use --review to submit by hand")
    from playwright.sync_api import sync_playwright  # lazy import: package is optional

    profile, voice = _load_profile(), _load_voice()
    resume_path = (profile.get("facts") or {}).get("resume_path", "")
    rows = apply_queue.load_queue()
    todo = [r for r in rows if r.get("status") == "queued" and r.get("ats") == "greenhouse"]
    if limit:
        todo = todo[:limit]
    if not todo:
        log.info("no queued greenhouse rows to process")
        return

    with sync_playwright() as p:
        # --review needs a visible window so the human can review + submit.
        browser = p.chromium.launch(headless=not (headful or review))
        ctx = browser.new_context()
        stop = False
        for row in todo:
            if stop:
                break
            page = ctx.new_page()
            try:
                page.goto(row["url"], timeout=30000)
                _reach_form(page)
                decisions = [_apply_field(page, f, profile, voice, resume_path)
                             for f in extract_fields(page)]
                shot = _shot(page, row["id"])
                status = "prepared"
                if review:
                    status = _hold_for_review(row)   # blocks; window stays open
                    if status == "__quit__":
                        status, stop = "prepared", True
                rows = apply_queue.update_row(rows, row["id"], status=status,
                                              screenshot=shot, note=_note(decisions),
                                              attempts=(row.get("attempts") or 0) + 1)
                log.info("%s %s -> %s", status, row["company"], shot)
            except Exception as exc:  # noqa: BLE001 — a bad row fails soft, batch continues
                rows = apply_queue.update_row(rows, row["id"], status="failed",
                                              note=f"driver error: {exc}"[:200])
                log.error("failed %s: %s", row.get("company"), exc)
            finally:
                apply_queue.save_queue(rows)
                page.close()
        browser.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Auto-apply Playwright driver (Greenhouse, shadow mode)")
    ap.add_argument("--limit", type=int, default=None, help="max queued rows to process")
    ap.add_argument("--headful", action="store_true", help="show the browser window")
    ap.add_argument("--review", action="store_true",
                    help="headful: fill each form, then pause so you review + click Submit yourself")
    ap.add_argument("--submit", action="store_true", help="(reserved for 4c; currently inert)")
    args = ap.parse_args()
    run(limit=args.limit, headful=args.headful, submit=args.submit, review=args.review)


if __name__ == "__main__":
    main()
