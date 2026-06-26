# Assisted Apply - Claude-in-Chrome Playbook

On-demand, human-in-the-loop. Run when Arnav picks a listing (from the email or
`data/listings.csv`) and says "apply to this." **Never auto-submits.**

Inputs: a listing URL + `profile/profile.yml` (the `facts` section).

## Procedure

1. **Open** the listing URL in a new Chrome tab (`tabs_create_mcp` → `navigate`).
2. **Reach the form.** If it's a job-description page, click "Apply" / "Apply Now".
   JobRight links route through jobright.ai first - follow through to the real application.
3. **Read the form** (`read_page`) and enumerate every field + its label.
4. **Autofill the deterministic 80%** from `profile.facts` using the field map below.
5. **STOP and hand back** for anything in "Always leave for the human" - do not guess.
6. **Summarize**: list what was filled, what's left, and any uncertain mappings. Wait for
   Arnav to review and click submit himself.

## Field map (profile.facts → common ATS labels)

| profile.facts | Matches labels like |
|---|---|
| `full_name` | Full name / First + Last (split on space) |
| `email` | Email |
| `phone` | Phone / Mobile |
| `location` | Location / City / Current location |
| `school` | School / University / Institution |
| `degree` | Degree |
| `grad_month_year` | Graduation date / Expected graduation |
| `gpa` | GPA |
| `links.github` | GitHub / Portfolio (if no dedicated portfolio field) |
| `links.linkedin` | LinkedIn / LinkedIn URL |
| `links.portfolio` | Website / Portfolio |
| `resume_path` | Resume / CV upload (`file_upload`) |
| `work_authorization.us_authorized` | "Are you authorized to work in the US?" |
| `work_authorization.needs_sponsorship` | "Will you require sponsorship?" |

ATS-specific notes:
- **Greenhouse / Lever / Ashby:** simple labeled inputs - high autofill success.
- **Workday:** multi-step, often requires creating an account first → pause for Arnav to
  log in / create the account, then resume autofill.

## Always leave for the human (never guess)

- Free-text / essay questions ("Why this company?", "Describe a project") - flag for Arnav.
  *(Drafting these is the shelved LaTeX-aware bullets feature - not active yet.)*
- EEO / demographics (race, gender, veteran, disability) - decline-to-answer is fine; his call.
- Anything where the label↔field mapping is ambiguous.
- The final **Submit** click. Always.

## Safety
- One application at a time. Don't open multiple company forms in parallel.
- Avoid clicking anything that triggers a browser dialog (alert/confirm) - it freezes the tab.
- If the same company has multiple roles, confirm with Arnav before applying to more than one.
