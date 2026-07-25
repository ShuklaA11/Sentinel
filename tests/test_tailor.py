"""Unit tests for tailor's pure helpers. No network, no tectonic — the LLM rewrite and
compile paths are integration-only; here we cover the deterministic logic."""
from src import tailor


def test_count_pages_prefers_page_tree_count():
    assert tailor._count_pages(b"<< /Type /Pages /Count 2 /Kids [1 0 R] >>") == 2


def test_count_pages_falls_back_to_page_objects():
    data = b"/Type /Page /MediaBox [0 0 612 792]  /Type /Page/Contents"
    assert tailor._count_pages(data) == 2


def test_count_pages_never_returns_zero():
    assert tailor._count_pages(b"no pdf structure here") == 1


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


# --- LLM wiring: _rewrite / _tighten now route through llm.complete ----------

def test_rewrite_prompt_carries_truthfulness_guard_and_applies_completion(monkeypatch):
    """The rewrite prompt keeps its truthfulness guard, and a canned INDEX|||BULLET
    completion is applied through _apply_rewrites."""
    captured = {}

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
    captured = {}

    def fake_complete(prompt, *, max_tokens=2048):
        captured["prompt"] = prompt
        return "0|||short bullet"

    monkeypatch.setattr(tailor.llm, "complete", fake_complete)
    out = tailor._tighten(["a rather long bullet"], "Co", "CV Intern", ["2 pages (resume must be 1)"])

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


# --- full wiring: tailor() returns a path ------------------------------------

def _profile_load(tex_path: str, base_pdf: str = "/base/resume.pdf"):
    """Fake _load_yaml: profile carries facts, resume_bank is empty."""
    def fake_load(path):
        if "profile" in path:
            return {"facts": {"resume_tex_path": tex_path, "resume_path": base_pdf}}
        return {}
    return fake_load


def test_tailor_returns_tailored_pdf_on_success(tmp_path, monkeypatch):
    tex = tmp_path / "base.tex"
    tex.write_text("\\begin{document}\\resumeItem{old bullet}\\end{document}")
    monkeypatch.setattr(tailor, "_load_yaml", _profile_load(str(tex)))
    monkeypatch.setattr(tailor.llm, "complete", lambda prompt, *, max_tokens=2048: "0|||new bullet")
    monkeypatch.setattr(tailor, "OUT_DIR", str(tmp_path / "tailored"))
    out_pdf = str(tmp_path / "out.pdf")
    monkeypatch.setattr(tailor, "_compile_inspect", lambda p: (out_pdf, []))

    assert tailor.tailor("Anthropic", "ML Intern", "startup", "") == out_pdf


def test_tailor_returns_pdf_on_residual_layout_issue(tmp_path, monkeypatch):
    tex = tmp_path / "base.tex"
    tex.write_text("\\begin{document}\\resumeItem{old bullet}\\end{document}")
    monkeypatch.setattr(tailor, "_load_yaml", _profile_load(str(tex)))
    monkeypatch.setattr(tailor.llm, "complete", lambda prompt, *, max_tokens=2048: "0|||new bullet")
    monkeypatch.setattr(tailor, "OUT_DIR", str(tmp_path / "tailored"))
    out_pdf = str(tmp_path / "out.pdf")
    # never resolves the layout issue -> after the retry loop, keep best-effort pdf
    monkeypatch.setattr(tailor, "_compile_inspect", lambda p: (out_pdf, ["2 pages (resume must be 1)"]))

    assert tailor.tailor("Co", "CV Intern", "startup", "") == out_pdf


def test_tailor_returns_base_resume_when_fact_gate_fails(monkeypatch):
    # base .tex missing -> precondition (fact) gate fails -> fall back to base resume path
    monkeypatch.setattr(tailor, "_load_yaml", _profile_load("/does/not/exist.tex"))
    assert tailor.tailor("Co", "Title", "startup", "") == "/base/resume.pdf"


def test_tailor_returns_base_resume_when_compile_unavailable(tmp_path, monkeypatch):
    tex = tmp_path / "base.tex"
    tex.write_text("\\begin{document}\\resumeItem{old bullet}\\end{document}")
    monkeypatch.setattr(tailor, "_load_yaml", _profile_load(str(tex)))
    monkeypatch.setattr(tailor.llm, "complete", lambda prompt, *, max_tokens=2048: "0|||new bullet")
    monkeypatch.setattr(tailor, "OUT_DIR", str(tmp_path / "tailored"))
    monkeypatch.setattr(tailor, "_compile_inspect", lambda p: (None, []))

    assert tailor.tailor("Co", "Title", "startup", "") == "/base/resume.pdf"
