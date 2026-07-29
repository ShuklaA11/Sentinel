"""Unit tests for tailor's pure helpers and wiring. No network, no tectonic — the LLM
rewrite and compile paths are stubbed; here we cover the deterministic logic, the LLM
routing through llm.complete, the keyword-injection prompt, the plausibility-gated
headline, and tailor()'s return contract."""
from src import tailor


def test_count_pages_prefers_page_tree_count():
    assert tailor._count_pages(b"<< /Type /Pages /Count 2 /Kids [1 0 R] >>") == 2


def test_count_pages_falls_back_to_page_objects():
    data = b"/Type /Page /MediaBox [0 0 612 792]  /Type /Page/Contents"
    assert tailor._count_pages(data) == 2


def test_count_pages_never_returns_zero():
    assert tailor._count_pages(b"no pdf structure here") == 1


def test_count_pages_reads_compressed_pdf():
    """Regression: tectonic emits compressed object streams, so the page tree is NOT
    visible as plaintext '/Count N' or '/Type /Page'. The byte-regex fell back to 1 and
    silently shipped multi-page resumes as '✓ compiled'. This fixture is a real
    tectonic-produced 2-page PDF; it must count as 2, not 1."""
    import os
    import re

    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "two_page_compressed.pdf")
    data = open(fixture, "rb").read()
    # the byte-regex the old implementation relied on finds nothing here...
    assert not re.findall(rb"/Count\s+(\d+)", data)
    assert not re.findall(rb"/Type\s*/Page[^s]", data)
    # ...yet the true page count is 2.
    assert tailor._count_pages(data) == 2


def test_render_swaps_only_resumeitem_bodies():
    tex = "\\begin{document}\\resumeItem{old one}\\resumeItem{old two}\\end{document}"
    items = tailor._extract_items(tex)
    out = tailor._render(tex, items, ["NEW ONE", "NEW TWO"])
    assert "\\resumeItem{NEW ONE}" in out
    assert "\\resumeItem{NEW TWO}" in out
    assert "old one" not in out and "old two" not in out


def test_apply_rewrites_keeps_original_for_missing_and_unsafe_rows():
    bullets = ["first bullet", "second bullet", "third bullet"]
    # row 1 rewritten; row 2 unsafe (unbalanced brace) -> original kept; row 0 absent -> original
    text = "1|||second improved\n2|||broken { brace"
    out = tailor._apply_rewrites(text, bullets)
    assert out == ["first bullet", "second improved", "third bullet"]


def test_safe_bullet_rejects_overlong_and_unescaped_specials():
    assert tailor._safe_bullet("x" * 200, "orig") == "orig"          # too long
    assert tailor._safe_bullet("cut 50% of cost", "orig") == "orig"  # unescaped %
    assert tailor._safe_bullet("cut 50\\% of cost", "orig") == "cut 50\\% of cost"


def test_xetex_compat_comments_out_pdftex_only_lines():
    tex = "\\input{glyphtounicode}\n\\pdfgentounicode=1\n"
    out = tailor._xetex_compat(tex)
    assert "%\\input{glyphtounicode}" in out
    assert "%\\pdfgentounicode=1" in out


# --- LLM wiring: _rewrite / _tighten route through llm.complete --------------
# The completion helper is monkeypatched so nothing hits the network; we capture the
# prompt string each pass would send and feed back a canned INDEX|||BULLET response.

def test_rewrite_prompt_carries_truthfulness_guard_and_applies_completion(monkeypatch):
    """The rewrite prompt keeps its truthfulness guard byte-for-byte, and a canned
    INDEX|||BULLET completion is applied through _apply_rewrites."""
    captured: dict = {}

    def fake_complete(prompt, *, max_tokens=2048):
        captured["prompt"] = prompt
        captured["max_tokens"] = max_tokens
        return "0|||Shipped YOLOv26 detector on-device"

    monkeypatch.setattr(tailor.llm, "complete", fake_complete)
    out = tailor._rewrite(["old bullet"], "Anthropic", "ML Intern", "impact", "", {})

    p = captured["prompt"]
    assert "TRUTHFUL: only reword / re-emphasize" in p
    assert "CHARACTER-FOR-CHARACTER" in p
    assert "YOLOv26" in p  # anti-'correction' guard survives byte-for-byte
    assert captured["max_tokens"] == 2048
    assert out == ["Shipped YOLOv26 detector on-device"]


def test_tighten_prompt_carries_guard_and_applies_completion(monkeypatch):
    captured: dict = {}

    def fake_complete(prompt, *, max_tokens=2048):
        captured["prompt"] = prompt
        return "0|||short bullet"

    monkeypatch.setattr(tailor.llm, "complete", fake_complete)
    out = tailor._tighten(["a rather long bullet"], "Co", "CV Intern",
                          ["2 pages (resume must be 1)"])

    assert "CHARACTER-FOR-CHARACTER" in captured["prompt"]
    assert out == ["short bullet"]


def test_rewrite_falls_back_to_originals_when_no_completion(monkeypatch):
    monkeypatch.setattr(tailor.llm, "complete", lambda prompt, *, max_tokens=2048: None)
    bullets = ["keep me exactly"]
    assert tailor._rewrite(bullets, "Co", "Title", "e", "", {}) == bullets


def test_tighten_keeps_current_when_no_completion(monkeypatch):
    monkeypatch.setattr(tailor.llm, "complete", lambda prompt, *, max_tokens=2048: None)
    bullets = ["keep me exactly"]
    assert tailor._tighten(bullets, "Co", "Title", ["x"]) == bullets


# --- keyword injection into the rewrite prompt -------------------------------
# The truthful keyword-coverage nudge is appended AFTER the HARD RULES; an empty
# candidate list adds no block at all. Both paths route through llm.complete.

def test_rewrite_prompt_includes_injection_keywords(monkeypatch):
    captured: dict = {}

    def fake_complete(prompt, *, max_tokens=2048):
        captured["prompt"] = prompt
        return "0|||reworded bullet"

    monkeypatch.setattr(tailor.llm, "complete", fake_complete)
    tailor._rewrite(["built models"], "Anthropic", "ML Intern", "emphasis", "jd",
                    {}, inject=["pytorch", "kubernetes"])

    assert "employer keywords the candidate genuinely has" in captured["prompt"]
    assert "pytorch" in captured["prompt"]
    assert "kubernetes" in captured["prompt"]


def test_rewrite_prompt_omits_injection_block_when_empty(monkeypatch):
    captured: dict = {}

    def fake_complete(prompt, *, max_tokens=2048):
        captured["prompt"] = prompt
        return "0|||reworded bullet"

    monkeypatch.setattr(tailor.llm, "complete", fake_complete)
    tailor._rewrite(["built models"], "Anthropic", "ML Intern", "e", "", {}, inject=[])

    assert "employer keywords the candidate genuinely has" not in captured["prompt"]
    # the truthfulness guard is always present, injection or not.
    assert "TRUTHFUL" in captured["prompt"]


# --- target-title headline (plausibility gate + LaTeX safety) ----------------

_HEADER_TEX = (
    "\\begin{document}\n"
    "\\begin{center}\n"
    "    \\textbf{\\Huge \\scshape Arnav Shukla} \\\\ \\vspace{1pt}\n"
    "    \\small email $|$ github\n"
    "\\end{center}\n"
    "\\resumeSubheading{Research Assistant}{2025}{Lab}{City}\n"
    "\\resumeItem{trained a model}\n"
    "\\end{document}\n"
)


def test_role_title_headline_is_opt_in_by_default():
    """The target-title headline is opt-in: a tagline under the name is a weak ATS
    signal for a student resume (ATS read titles from the experience section), so
    tailor() must default role_title to False."""
    import inspect

    assert inspect.signature(tailor.tailor).parameters["role_title"].default is False


def test_headline_injected_for_track_matching_title():
    profile = {"preferences": {"tracks": ["ml"]}}

    headline = tailor._plan_headline(_HEADER_TEX, "Machine Learning Intern", profile)
    assert headline == "Machine Learning Intern"

    out = tailor._inject_headline(_HEADER_TEX, headline)
    # headline appears under the name...
    assert "\\textbf{\\large Machine Learning Intern}" in out
    # ...the real name line stays byte-identical...
    assert "\\textbf{\\Huge \\scshape Arnav Shukla} \\\\ \\vspace{1pt}" in out
    # ...and the real experience subheading is never rewritten.
    assert "\\resumeSubheading{Research Assistant}{2025}{Lab}{City}" in out


def test_headline_injected_for_computer_vision_title():
    # Broadened DEFAULT_TRACK_TERMS covers the vision/perception core, so a CV title
    # passes the plausibility gate even with an empty profile tracks list.
    profile = {"preferences": {"tracks": []}}
    headline = tailor._plan_headline(_HEADER_TEX, "Computer Vision Engineer Intern", profile)
    assert headline is not None
    assert headline == "Computer Vision Engineer Intern"


def test_headline_skipped_for_track_mismatched_title():
    profile = {"preferences": {"tracks": ["ml", "data"]}}
    assert tailor._plan_headline(_HEADER_TEX, "Warehouse Associate Intern", profile) is None


def test_headline_skipped_when_title_unsafe_after_escaping():
    # unbalanced brace cannot be made LaTeX-safe -> no injection even though 'data' matches.
    profile = {"preferences": {"tracks": []}}
    assert tailor._plan_headline(_HEADER_TEX, "Data { Intern", profile) is None


def test_headline_escapes_latex_specials():
    profile = {"preferences": {"tracks": ["product"]}}
    headline = tailor._plan_headline(_HEADER_TEX, "Product & Growth Intern", profile)
    assert headline == "Product \\& Growth Intern"


# --- full tailor() wiring: coverage delta, fact gate, return contract --------

def _base_profile(tex_path: str) -> dict:
    return {
        "facts": {"resume_tex_path": tex_path, "resume_path": "/base/resume.pdf"},
        "preferences": {"tracks": ["ml"]},
        "skills": {"languages": ["Python"], "tools": ["Docker"]},
    }


def _write_tex(tmp_path, body_item: str) -> str:
    tex = (
        "\\begin{document}\n"
        "\\begin{center}\n"
        "    \\textbf{\\Huge \\scshape Arnav Shukla} \\\\ \\vspace{1pt}\n"
        "\\end{center}\n"
        f"\\resumeItem{{{body_item}}}\n"
        "\\end{document}\n"
    )
    path = tmp_path / "resume.tex"
    path.write_text(tex)
    return str(path)


def test_tailor_reports_coverage_delta(tmp_path, monkeypatch, capsys):
    tex_path = _write_tex(tmp_path, "built web services")
    profile = _base_profile(tex_path)
    monkeypatch.setattr(tailor, "_load_yaml",
                        lambda p: profile if p.endswith("profile.yml") else {})
    monkeypatch.setattr(tailor, "OUT_DIR", str(tmp_path / "out"))
    # rewrite surfaces the genuinely-held keywords python + docker.
    monkeypatch.setattr(tailor, "_rewrite",
                        lambda *a, **k: ["built Python services with Docker"])
    monkeypatch.setattr(tailor, "_compile_inspect", lambda p: (str(tmp_path / "r.pdf"), []))
    monkeypatch.setattr(tailor.verify_facts, "load_sources", lambda: [])

    tailor.tailor("Acme", "ML Intern", "startup", "Python and Docker")

    out = capsys.readouterr().out
    # base tex had neither keyword (0%), rewritten resume has both (100%), +2 gained.
    assert "JD keyword coverage: 0% -> 100% (+2 keywords)" in out


def test_tailor_fact_gate_falls_back_and_returns_base_resume(tmp_path, monkeypatch, capsys):
    tex_path = _write_tex(tmp_path, "optimized the pipeline")
    profile = _base_profile(tex_path)
    monkeypatch.setattr(tailor, "_load_yaml",
                        lambda p: profile if p.endswith("profile.yml") else {})
    monkeypatch.setattr(tailor, "OUT_DIR", str(tmp_path / "out"))
    # the rewrite fabricates a metric absent from the ground-truth sources.
    monkeypatch.setattr(tailor, "_rewrite",
                        lambda *a, **k: ["accelerated the pipeline 5x"])
    monkeypatch.setattr(tailor, "_compile_inspect", lambda p: (str(tmp_path / "r.pdf"), []))
    monkeypatch.setattr(tailor.verify_facts, "load_sources", lambda: ["no numbers here"])

    result = tailor.tailor("Acme", "ML Intern", "startup", "")

    out = capsys.readouterr().out
    # fact gate blocks the tailored PDF, prints the fallback, and returns the base path.
    assert "fact gate failed" in out
    assert "5x" in out
    assert "/base/resume.pdf" in out
    assert "✓ compiled" not in out
    assert result == "/base/resume.pdf"


def test_tailor_returns_tailored_pdf_on_success(tmp_path, monkeypatch):
    tex_path = _write_tex(tmp_path, "old bullet")
    profile = _base_profile(tex_path)
    monkeypatch.setattr(tailor, "_load_yaml",
                        lambda p: profile if p.endswith("profile.yml") else {})
    monkeypatch.setattr(tailor.llm, "complete", lambda prompt, *, max_tokens=2048: "0|||new bullet")
    monkeypatch.setattr(tailor, "OUT_DIR", str(tmp_path / "tailored"))
    monkeypatch.setattr(tailor.verify_facts, "load_sources", lambda: [])
    out_pdf = str(tmp_path / "out.pdf")
    monkeypatch.setattr(tailor, "_compile_inspect", lambda p: (out_pdf, []))

    assert tailor.tailor("Anthropic", "ML Intern", "startup", "") == out_pdf


def test_tailor_returns_pdf_on_residual_layout_issue(tmp_path, monkeypatch):
    tex_path = _write_tex(tmp_path, "old bullet")
    profile = _base_profile(tex_path)
    monkeypatch.setattr(tailor, "_load_yaml",
                        lambda p: profile if p.endswith("profile.yml") else {})
    monkeypatch.setattr(tailor.llm, "complete", lambda prompt, *, max_tokens=2048: "0|||new bullet")
    monkeypatch.setattr(tailor, "OUT_DIR", str(tmp_path / "tailored"))
    monkeypatch.setattr(tailor.verify_facts, "load_sources", lambda: [])
    out_pdf = str(tmp_path / "out.pdf")
    # never resolves the layout issue -> after the retry loop, keep best-effort pdf.
    monkeypatch.setattr(tailor, "_compile_inspect",
                        lambda p: (out_pdf, ["2 pages (resume must be 1)"]))

    assert tailor.tailor("Co", "CV Intern", "startup", "") == out_pdf


def test_tailor_returns_base_resume_when_base_tex_missing(monkeypatch):
    # base .tex missing -> precondition fails before any rewrite -> return base resume path.
    profile = _base_profile("/does/not/exist.tex")
    monkeypatch.setattr(tailor, "_load_yaml",
                        lambda p: profile if p.endswith("profile.yml") else {})
    assert tailor.tailor("Co", "Title", "startup", "") == "/base/resume.pdf"


def test_tailor_returns_base_resume_when_compile_unavailable(tmp_path, monkeypatch):
    tex_path = _write_tex(tmp_path, "old bullet")
    profile = _base_profile(tex_path)
    monkeypatch.setattr(tailor, "_load_yaml",
                        lambda p: profile if p.endswith("profile.yml") else {})
    monkeypatch.setattr(tailor.llm, "complete", lambda prompt, *, max_tokens=2048: "0|||new bullet")
    monkeypatch.setattr(tailor, "OUT_DIR", str(tmp_path / "tailored"))
    monkeypatch.setattr(tailor.verify_facts, "load_sources", lambda: [])
    monkeypatch.setattr(tailor, "_compile_inspect", lambda p: (None, []))

    assert tailor.tailor("Co", "Title", "startup", "") == "/base/resume.pdf"


def test_tailor_titles_is_opt_in_by_default():
    import inspect
    assert inspect.signature(tailor.tailor).parameters["tailor_titles"].default is False


def _write_tex_with_subheading(tmp_path) -> str:
    tex = (
        "\\begin{document}\n"
        "\\begin{center}\n"
        "    \\textbf{\\Huge \\scshape Arnav Shukla} \\\\ \\vspace{1pt}\n"
        "\\end{center}\n"
        "\\resumeSubheading\n"
        "    {INSPIRE Lab}{January 2026 -- Present}\n"
        "    {Undergraduate Researcher}{Austin, TX}\n"
        "\\resumeItem{Built a YOLO26 pipeline detecting robotaxis}\n"
        "\\end{document}\n"
    )
    path = tmp_path / "resume.tex"
    path.write_text(tex)
    return str(path)


def _title_profile(tex_path: str) -> dict:
    profile = _base_profile(tex_path)
    profile["role_titles"] = {
        "INSPIRE Lab": ["Undergraduate Researcher", "Computer Vision Researcher"]
    }
    return profile


def test_tailor_titles_swaps_approved_alternate(tmp_path, monkeypatch, capsys):
    tex_path = _write_tex_with_subheading(tmp_path)
    profile = _title_profile(tex_path)
    monkeypatch.setattr(tailor, "_load_yaml", lambda p: profile if p.endswith("profile.yml") else {})
    monkeypatch.setattr(tailor, "OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(tailor, "_rewrite", lambda *a, **k: ["Built a YOLO26 pipeline detecting robotaxis"])
    monkeypatch.setattr(tailor, "_compile_inspect", lambda p: (str(tmp_path / "r.pdf"), []))
    monkeypatch.setattr(tailor.verify_facts, "load_sources", lambda: [])

    tailor.tailor("Waymo", "Computer Vision Intern", "startup",
                  "computer vision perception object detection", tailor_titles=True)

    out = capsys.readouterr().out
    assert "TITLE CHANGES" in out
    assert "Undergraduate Researcher  ->  Computer Vision Researcher" in out
    written = (tmp_path / "out" / "waymo_computer_vision_intern.tex").read_text()
    assert "Computer Vision Researcher" in written
    assert "Undergraduate Researcher" not in written


def test_tailor_titles_off_by_default_keeps_titles(tmp_path, monkeypatch, capsys):
    tex_path = _write_tex_with_subheading(tmp_path)
    profile = _title_profile(tex_path)
    monkeypatch.setattr(tailor, "_load_yaml", lambda p: profile if p.endswith("profile.yml") else {})
    monkeypatch.setattr(tailor, "OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(tailor, "_rewrite", lambda *a, **k: ["Built a YOLO26 pipeline detecting robotaxis"])
    monkeypatch.setattr(tailor, "_compile_inspect", lambda p: (str(tmp_path / "r.pdf"), []))
    monkeypatch.setattr(tailor.verify_facts, "load_sources", lambda: [])

    tailor.tailor("Waymo", "Computer Vision Intern", "startup", "computer vision perception")

    written = (tmp_path / "out" / "waymo_computer_vision_intern.tex").read_text()
    assert "Undergraduate Researcher" in written
    assert "Computer Vision Researcher" not in written
    assert "TITLE CHANGES" not in capsys.readouterr().out


# --- bullet-pool selection wiring (opt-in, default off) ----------------------
# select_bullets applies resumeselect.select BEFORE bullet extraction: it picks the
# most JD-relevant subset of the pooled \resumeItem bullets (active AND commented)
# under a one-page budget, byte-preserving everything else. Off by default.

def test_select_bullets_is_opt_in_by_default():
    import inspect
    assert inspect.signature(tailor.tailor).parameters["select_bullets"].default is False


def _write_tex_with_pool(tmp_path) -> str:
    # One experience with a two-bullet pool: an off-topic ACTIVE default plus a commented
    # JD-relevant pool bullet. Budget = active count = 1, so a JD hit flips the pair.
    tex = (
        "\\begin{document}\n"
        "\\begin{center}\n"
        "    \\textbf{\\Huge \\scshape Arnav Shukla} \\\\ \\vspace{1pt}\n"
        "\\end{center}\n"
        "\\resumeSubheading\n"
        "    {INSPIRE Lab}{January 2026 -- Present}\n"
        "    {Undergraduate Researcher}{Austin, TX}\n"
        "\\resumeItem{Organized a campus bake sale for the student council}\n"
        "% \\resumeItem{Built a computer vision object detection pipeline}\n"
        "\\end{document}\n"
    )
    path = tmp_path / "resume.tex"
    path.write_text(tex)
    return str(path)


def test_select_bullets_activates_jd_relevant_pooled_bullet(tmp_path, monkeypatch, capsys):
    tex_path = _write_tex_with_pool(tmp_path)
    profile = _base_profile(tex_path)
    monkeypatch.setattr(tailor, "_load_yaml", lambda p: profile if p.endswith("profile.yml") else {})
    monkeypatch.setattr(tailor, "OUT_DIR", str(tmp_path / "out"))
    # identity rewrite over the (now newly-selected) active bullets.
    monkeypatch.setattr(tailor, "_rewrite", lambda bullets, *a, **k: list(bullets))
    monkeypatch.setattr(tailor, "_compile_inspect", lambda p: (str(tmp_path / "r.pdf"), []))
    monkeypatch.setattr(tailor.verify_facts, "load_sources", lambda: [])

    tailor.tailor("Waymo", "Computer Vision Intern", "startup",
                  "computer vision object detection perception model", select_bullets=True)

    out = capsys.readouterr().out
    assert "BULLETS SELECTED (fit to one page, by JD relevance):" in out
    assert "+activated" in out and "-deactivated" in out

    written = (tmp_path / "out" / "waymo_computer_vision_intern.tex").read_text()
    # the JD-relevant pool bullet is now ACTIVE (uncommented)...
    assert "\\resumeItem{Built a computer vision object detection pipeline}" in written
    assert "% \\resumeItem{Built a computer vision object detection pipeline}" not in written
    # ...and the off-topic default is DEACTIVATED (commented out).
    assert "% \\resumeItem{Organized a campus bake sale for the student council}" in written


def test_select_bullets_off_by_default_keeps_activation(tmp_path, monkeypatch, capsys):
    tex_path = _write_tex_with_pool(tmp_path)
    profile = _base_profile(tex_path)
    monkeypatch.setattr(tailor, "_load_yaml", lambda p: profile if p.endswith("profile.yml") else {})
    monkeypatch.setattr(tailor, "OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(tailor, "_rewrite", lambda bullets, *a, **k: list(bullets))
    monkeypatch.setattr(tailor, "_compile_inspect", lambda p: (str(tmp_path / "r.pdf"), []))
    monkeypatch.setattr(tailor.verify_facts, "load_sources", lambda: [])

    tailor.tailor("Waymo", "Computer Vision Intern", "startup",
                  "computer vision object detection perception model")

    written = (tmp_path / "out" / "waymo_computer_vision_intern.tex").read_text()
    # activation is UNCHANGED: the default stays active, the pool bullet stays commented.
    assert "\\resumeItem{Organized a campus bake sale for the student council}" in written
    assert "% \\resumeItem{Organized a campus bake sale for the student council}" not in written
    assert "% \\resumeItem{Built a computer vision object detection pipeline}" in written
    assert "BULLETS SELECTED" not in capsys.readouterr().out
