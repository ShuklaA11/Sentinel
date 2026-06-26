# Sentinel

Early-detection internship pipeline. Sentinel polls ~290 ATS endpoints and community job repos every 15 minutes, filters to internships in your tracks, fit-ranks new listings with Claude, and emails you a digest so you see roles within minutes of posting, not days. It also tailors your resume per role, reviews it recruiter-style, and builds company-specific interview prep.

**Core principle: be first, never break.** A dead company slug or a network blip must never crash a run, so every fetcher logs and returns empty. Scoring and email degrade gracefully: without an API key or SMTP creds, the core detect-and-alert loop still runs.

## How it works

The poll pipeline runs on a cron and is a straight line: fetch, filter, dedupe, rank, notify.

1. **Fetch** (`src/sources.py`) - threaded pulls from Greenhouse, Lever, Ashby, SmartRecruiters, and BambooHR endpoints, plus community application repos. Every listing is normalized to a common shape.
2. **Filter** (`src/filter.py`) - keeps only titles that look like internships, in a configured track (ML / SWE / Product), inside the target season window, with word-boundary matching so "intern" never hits "internal."
3. **Dedupe + persist** (`src/store.py`) - new listings are diffed against a committed seen-set; fresh hits append to a CSV tracker. Git is the database.
4. **Rank** (`src/rank.py`) - new listings are fit-scored 0 to 100 against your profile by Claude Haiku, in batches. Dealbreakers are penalized hard. Skipped cleanly if no key or profile.
5. **Notify** (`src/notify.py`) - an email digest grouped by track and sorted by score, via SMTP.

### Beyond detection

- **Slug discovery** (`src/harvest.py`) - a daily job mines new ATS slugs from community-repo application URLs into `config/harvested.yml`, so the poller hits those companies directly. Kept separate from polling so polls stay fast and the slug list stays reviewable in git.
- **Resume review** (`src/review.py`) - recruiter-grade scoring via Claude, reading the resume PDF natively, with a rubric re-weighted toward ML/research depth.
- **Resume tailoring** (`src/tailor.py`) - rewrites only the bullet text in a LaTeX template (preserving it byte-for-byte) and compiles to PDF with tectonic. Reword and re-emphasize real bullets only, never fabricate.
- **Interview prep** (`src/interview.py`) - per-company packs: DSA pulled live from a frequency-sorted LeetCode repo, a curated track-aware behavioral bank, and optional LLM questions about your specific projects.
- **Assisted apply** (`apply/autofill.md`) - a human-in-the-loop Claude-in-Chrome playbook that autofills the deterministic fields and stops for anything requiring judgment. Never auto-submits.

## Automation

Both jobs run free on GitHub Actions, credentials supplied as repo secrets:

| Workflow | Schedule | Does |
|---|---|---|
| `poll.yml` | every 15 min | fetch, filter, rank, email, commit new listings |
| `harvest.yml` | daily 07:00 UTC | mine new ATS slugs from community repos |

## Tech stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| HTTP | requests (threaded fetchers) |
| Config | YAML |
| Fit-ranking, review, tailoring | Anthropic Claude (Haiku for ranking, Sonnet for review/tailoring) |
| Resume compile | tectonic (LaTeX) |
| Email | SMTP (Gmail app password) |
| Scheduling | GitHub Actions cron |
| Storage | git-committed seen-set + CSV tracker |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # fill in keys; in CI, use repo Secrets instead
python -m src.run --dry-run
```

Configuration lives in `config/` (companies, filters, tracks, season window) and a gitignored `profile/profile.yml` (your background and preferences, used for ranking and tailoring).

## Acknowledgements

The resume-review rubric is ported from [interviewstreet/hiring-agent](https://github.com/interviewstreet/hiring-agent) (MIT), re-weighted for ML/AI intern profiles.
