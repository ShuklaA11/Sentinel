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
