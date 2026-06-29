"""Email a digest of new listings via SMTP.

Credentials come from env (set as GitHub Actions secrets), never from files:
    SMTP_HOST (default smtp.gmail.com)
    SMTP_PORT (default 465, SSL)
    SMTP_USER  — the sending Gmail address
    SMTP_PASS  — a Gmail App Password (not the account password)
    ALERT_TO   — where to send the digest (defaults to SMTP_USER)

If SMTP_USER/SMTP_PASS are unset (e.g. local dev), it logs and no-ops so runs don't fail.
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

log = logging.getLogger("notify")


def _digest_text(new: list[dict]) -> str:
    lines = [f"{len(new)} new internship listing(s):\n"]
    for track in sorted({l["track"] for l in new}):
        rows = [l for l in new if l["track"] == track]
        lines.append(f"\n=== {track.upper()} ({len(rows)}) ===")
        for l in sorted(rows, key=lambda x: (-(x.get("score") or 0), x["company"])):
            loc = f" · {l['location']}" if l["location"] else ""
            sc = f"[{l['score']}] " if l.get("score") != "" else ""
            reason = f"  — {l['fit_reason']}" if l.get("fit_reason") else ""
            lines.append(f"  {sc}{l['company']} — {l['title']}{loc}{reason}\n    {l['url']}")
    return "\n".join(lines)


def _send(subject: str, new: list[dict]) -> bool:
    """Email `new` under `subject`. Returns True if sent, False if skipped/failed."""
    if not new:
        return False
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    if not user or not password:
        log.warning("SMTP_USER/SMTP_PASS unset — skipping email (%d listings)", len(new))
        return False

    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    to_addr = os.environ.get("ALERT_TO", user)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content(_digest_text(new))

    try:
        with smtplib.SMTP_SSL(host, port, timeout=30) as smtp:
            smtp.login(user, password)
            smtp.send_message(msg)
    except Exception as exc:  # noqa: BLE001 — surface any SMTP failure, don't crash the run
        log.error("email send failed: %s", exc)
        return False
    log.info("emailed %d listings to %s — %s", len(new), to_addr, subject)
    return True


def send_digest(new: list[dict]) -> bool:
    """Regular digest of new (sub-threshold) listings."""
    return _send(f"[internships] {len(new)} new listing(s)", new)


def send_high_fit_alert(new: list[dict]) -> bool:
    """Immediate alert for high-fit listings (score >= high_fit_threshold)."""
    return _send(f"\U0001F525 {len(new)} high-fit internship(s) — apply now", new)
