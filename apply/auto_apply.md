# Auto-Apply Driver — Claude-in-Chrome Playbook (SHADOW MODE)

Queue-driven batch apply. Processes `data/apply_queue.csv` rows with `status=queued`.
This is the **proving phase**: it fills every field it confidently can, screenshots the
completed form, and **STOPS before submit**. It never clicks Submit. Live submission is a
later step gated behind an explicit `--submit` flag, enabled only after shadow runs check out.

Inputs: `data/apply_queue.csv`, `profile/profile.yml` (`facts` + `answer_bank`),
and the field mapper `src/apply_fill.py` (`plan_fields`).

## Per-row procedure

For each `queued` row (oldest first, **one at a time**):

1. **Open** `row.url` in a new Chrome tab (`tabs_create_mcp` → `navigate`).
2. **Reach the form.** If it's a job-description page, click "Apply" / "Apply Now".
   JobRight/board links route through an intermediary — follow through to the real form.
3. **Enumerate fields** (`read_page`): for every input capture `{label, type, required, options}`.
4. **Plan** the fills by passing that list to `apply_fill.plan_fields(fields, profile)`. It returns:
   - `fills` — type/select these. For `type=file`, upload `facts.resume_path` (`file_upload`).
     For a fill marked `directive: true` (salary, how-heard), read the JD range / dropdown options
     and apply the instruction — do not type the directive text verbatim.
   - `drafts` — free-text questions. For each, call
     `essays.draft_answer(question, profile, voice, jd_text=<the role's JD>, word_limit=<any stated limit>)`
     where `voice` is the text of `profile/voice.md`. Fill the returned answer. It is already
     em-dash-free and grounded only in profile facts. If it returns `None` (no API key), treat
     that field as `needs_human` instead.
   - `skips` — leave blank (intentional, e.g. GPA declined).
   - `needs_human` — **do not fill.** These are unmapped or no-answer-on-file fields.
5. **Screenshot** the completed form (`computer` screenshot) → save under `data/apply_shots/`.
6. **Record** via `src.apply_queue`: `rows = load_queue()`,
   `update_row(rows, row.id, status="prepared", prepared_at=<now>, screenshot=<path>,`
   `attempts=<n+1>, note=<any needs_human labels + a "drafted: <labels>" marker>)`, then
   `save_queue(rows)`. Flagging which fields were LLM-drafted tells the reviewer exactly
   what to read closely.
7. **Move on.** Do not open the next form until this one is recorded.

## Shadow rule (non-negotiable in this mode)

- **Never click Submit / Apply-final / Send.** Stop at the completed, unsubmitted form.
- The point is to verify the mapper's field matches and eligibility answers on *real* forms
  before any autonomous submission exists. Report anything the mapper got wrong.
- **LLM-drafted essays get the closest review.** They are the highest-quality-risk output.
  In shadow mode the human reads every drafted answer before it would ever be submitted.

## Per-ATS notes

- **Greenhouse / Lever / Ashby:** single labeled forms — high fill rate, no account. Primary target.
- **SmartRecruiters / Workable / BambooHR:** mostly single-page; similar handling.
- **Workday:** account creation + multi-step + frequent CAPTCHA. Do **not** create accounts or solve
  CAPTCHAs autonomously. If a login/account/CAPTCHA wall blocks the form, set
  `status="needs_human"` with a note and move on — it can't be shadow-filled.
- **`ats=unknown` (repo-sourced):** resolve the real ATS after opening the URL; handle per above.

## Safety

- One application at a time. Never open multiple company forms in parallel.
- Avoid clicking anything that triggers a browser dialog (alert/confirm) — it freezes the tab.
- Pace politely; there is no rush in shadow mode.
- If a row can't be reached or errors out, set `status="failed"` with the reason and continue —
  a single bad row must never abort the batch.
