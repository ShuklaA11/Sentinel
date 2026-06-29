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

import html
import logging
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

TRACK_ORDER = ["ml", "swe", "product"]
SOURCE_LABELS = {
    "greenhouse": "Greenhouse", "lever": "Lever", "ashby": "Ashby",
    "smartrecruiters": "SmartRecruiters", "workable": "Workable",
    "workday": "Workday", "bamboohr": "BambooHR",
}


def _source_label(source: str) -> str:
    """Human label for a source: 'greenhouse' -> 'Greenhouse', 'repo:SimplifyJobs/..' -> 'SimplifyJobs'."""
    if not source:
        return ""
    if source.startswith("repo:"):
        return source[len("repo:"):].split("/")[0]
    return SOURCE_LABELS.get(source, source.title())


def _posted_label(posted_at: str, now: datetime | None = None) -> str:
    """Relative age from an ISO timestamp: 'posted today' / 'posted 3d ago'. '' if unparseable."""
    if not posted_at:
        return ""
    try:
        dt = datetime.fromisoformat(str(posted_at))
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    days = (now - dt).days
    if days <= 0:
        return "posted today"
    return "posted 1d ago" if days == 1 else f"posted {days}d ago"

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


def _badge(score) -> str:
    """A pill showing the fit score, color-coded by strength. Empty for unscored."""
    if not isinstance(score, int):
        return ""
    color = "#16a34a" if score >= 85 else "#d97706" if score >= 70 else "#6b7280"
    return (f'<span style="display:inline-block;padding:2px 9px;border-radius:999px;'
            f'font-size:12px;font-weight:700;color:#ffffff;background:{color};">{score}</span>')


def _digest_html(new: list[dict], high_fit: bool = False) -> str:
    """HTML version of the digest: inline styles + a table for alignment (email-safe)."""
    accent = "#dc2626" if high_fit else "#2563eb"
    noun = "internship" if len(new) == 1 else "internships"
    title = f"\U0001F525 {len(new)} high-fit {noun}" if high_fit else f"{len(new)} new {noun}"
    e = lambda s: html.escape(str(s or ""))

    tracks = {l["track"] for l in new}
    ordered = [t for t in TRACK_ORDER if t in tracks] + sorted(tracks - set(TRACK_ORDER))

    cards = []
    for track in ordered:
        rows = sorted([l for l in new if l["track"] == track],
                      key=lambda x: (-(x.get("score") or 0), x["company"]))
        cards.append(f'<div style="font-size:12px;font-weight:700;letter-spacing:.06em;'
                     f'color:#9ca3af;text-transform:uppercase;margin:18px 0 8px;">{e(track)} · {len(rows)}</div>')
        for l in rows:
            loc = (f'<div style="font-size:12px;color:#6b7280;margin-top:3px;">\U0001F4CD {e(l["location"])}</div>'
                   if l.get("location") else "")
            meta_bits = [b for b in (_source_label(l.get("source", "")),
                                     _posted_label(l.get("posted_at", ""))) if b]
            meta = (f'<div style="font-size:11px;color:#9ca3af;margin-top:4px;">{e(" · ".join(meta_bits))}</div>'
                    if meta_bits else "")
            reason = (f'<div style="font-size:12px;color:#6b7280;font-style:italic;margin-top:4px;">{e(l["fit_reason"])}</div>'
                      if l.get("fit_reason") else "")
            cards.append(
                f'<div style="border:1px solid #e5e7eb;border-radius:10px;padding:13px 15px;margin:9px 0;">'
                f'<table width="100%" cellpadding="0" cellspacing="0"><tr>'
                f'<td style="font-size:15px;font-weight:700;color:#111827;">{e(l["company"])}</td>'
                f'<td align="right">{_badge(l.get("score"))}</td></tr></table>'
                f'<div style="font-size:14px;color:#374151;margin-top:3px;">{e(l["title"])}</div>'
                f'{loc}{meta}{reason}'
                f'<a href="{html.escape(str(l.get("url", "")), quote=True)}" '
                f'style="display:inline-block;margin-top:10px;font-size:13px;font-weight:600;'
                f'color:{accent};text-decoration:none;">Apply &rarr;</a></div>')

    return (
        f'<div style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;'
        f'max-width:600px;margin:0 auto;background:#ffffff;color:#1f2937;">'
        f'<div style="padding:20px 24px;border-bottom:2px solid {accent};">'
        f'<div style="font-size:20px;font-weight:800;color:#111827;">{title}</div>'
        f'<div style="font-size:13px;color:#6b7280;margin-top:2px;">via Sentinel · ranked by fit</div></div>'
        f'<div style="padding:4px 24px 20px;">{"".join(cards)}</div>'
        f'<div style="padding:14px 24px;font-size:11px;color:#9ca3af;border-top:1px solid #f3f4f6;">'
        f'Sentinel — automated internship detection.</div></div>')


def _send(subject: str, new: list[dict], high_fit: bool = False) -> bool:
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
    msg.set_content(_digest_text(new))                                  # plain-text fallback
    msg.add_alternative(_digest_html(new, high_fit), subtype="html")    # rich version

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
    return _send(f"\U0001F525 {len(new)} high-fit internship(s) — apply now", new, high_fit=True)
