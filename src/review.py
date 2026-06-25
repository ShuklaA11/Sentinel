"""Phase 5: recruiter-grade resume review via Claude, tuned for ML/AI interns.

Ports the scoring rubric + fairness rules from interviewstreet/hiring-agent (MIT),
re-weighted away from open-source toward ML/research depth. Claude reads the resume PDF
natively and is enriched with public GitHub repo signals.

Run: python -m src.review            (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import base64
import json
import logging
import os

import requests
import yaml

log = logging.getLogger("review")

MODEL = os.environ.get("REVIEW_MODEL", "claude-sonnet-4-6")
PROFILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profile", "profile.yml")

# Categories (max points) — re-weighted for ML/AI interns vs hiring-agent's OSS-heavy rubric.
CATEGORIES = [
    ("research_and_ml_depth", 35, "Complexity, novelty, and rigor of ML/AI projects and research output"),
    ("projects_impact", 25, "Real-world impact, measurable results, and deployment of projects"),
    ("production_experience", 25, "Internships/work: production contributions, scale, engineering maturity"),
    ("technical_skills", 15, "Breadth and depth of a relevant ML/AI + SWE stack"),
]

# Ported verbatim in spirit from hiring-agent's CRITICAL FAIRNESS REQUIREMENTS.
SYSTEM = (
    "You are a senior ML hiring manager screening a resume for an ML/AI internship. "
    "Be strict, specific, and evidence-based.\n\n"
    "FAIRNESS — scores MUST NEVER depend on: the candidate's name, gender or demographics; "
    "college/university name; GPA or grades; city or location. Evaluate ONLY technical skills, "
    "project complexity and impact, research output, production experience, and communication."
)


def _load_profile() -> dict:
    with open(PROFILE_PATH) as f:
        return yaml.safe_load(f) or {}


def _github_repos(github_url: str) -> list[dict]:
    """Public repo signals (name, desc, language, stars, fork) — no auth needed."""
    if not github_url or "github.com/" not in github_url:
        return []
    user = github_url.rstrip("/").split("github.com/")[-1].split("/")[0]
    try:
        resp = requests.get(f"https://api.github.com/users/{user}/repos?per_page=100&sort=updated",
                            headers={"User-Agent": "job-applier/0.1"}, timeout=20)
        if resp.status_code != 200:
            log.warning("github HTTP %s for %s", resp.status_code, user)
            return []
        return [{"name": r["name"], "description": r.get("description"),
                 "language": r.get("language"), "stars": r.get("stargazers_count"),
                 "fork": r.get("fork")} for r in resp.json()]
    except requests.RequestException as exc:
        log.warning("github fetch failed: %s", exc)
        return []


def _rubric_text() -> str:
    cats = "\n".join(f"- {name} (0–{mx}): {desc}" for name, mx, desc in CATEGORIES)
    return (
        f"Score the attached resume on these categories (total 100):\n{cats}\n\n"
        "For each category give: score, 1–2 lines of evidence from the resume, and 1–3 concrete "
        "fixes that would raise it. Then give bonus points (notable strengths), deductions "
        "(red flags), and the 3–5 highest-leverage fixes overall.\n\n"
        'Return ONLY JSON: {"overall": int, "categories": [{"name": str, "score": int, '
        '"max": int, "evidence": str, "fixes": [str]}], "bonus": [str], "deductions": [str], '
        '"top_fixes": [str]}'
    )


def _client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed — run: pip install anthropic")
        return None
    return anthropic.Anthropic(api_key=key)


def run() -> dict | None:
    client = _client()
    if client is None:
        print("ANTHROPIC_API_KEY not set (or anthropic not installed) — cannot run review.")
        return None

    profile = _load_profile()
    resume_path = (profile.get("facts") or {}).get("resume_path", "")
    if not resume_path or not os.path.exists(resume_path):
        print(f"resume not found at: {resume_path!r} — set facts.resume_path in profile.yml")
        return None

    with open(resume_path, "rb") as f:
        pdf_b64 = base64.standard_b64encode(f.read()).decode()
    github = _github_repos((profile.get("facts", {}).get("links") or {}).get("github", ""))

    content = [
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64}},
        {"type": "text", "text": f"=== GITHUB DATA ===\n{json.dumps(github)[:4000]}\n\n{_rubric_text()}"},
    ]
    resp = client.messages.create(
        model=MODEL, max_tokens=2048, system=SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    text = resp.content[0].text.strip()
    text = text[text.find("{"): text.rfind("}") + 1]
    review = json.loads(text)
    _print_report(review)
    return review


def _print_report(r: dict) -> None:
    print(f"\n{'='*60}\n  RESUME REVIEW — overall {r.get('overall')}/100\n{'='*60}")
    for c in r.get("categories", []):
        bar = "█" * round(12 * c["score"] / max(c["max"], 1))
        print(f"\n  {c['name']:<22} {c['score']:>2}/{c['max']:<3} {bar}")
        print(f"    evidence: {c.get('evidence','')}")
        for fix in c.get("fixes", []):
            print(f"    fix: {fix}")
    if r.get("bonus"):
        print("\n  + bonus:", "; ".join(r["bonus"]))
    if r.get("deductions"):
        print("  - deductions:", "; ".join(r["deductions"]))
    print("\n  TOP FIXES:")
    for i, fix in enumerate(r.get("top_fixes", []), 1):
        print(f"    {i}. {fix}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    run()
