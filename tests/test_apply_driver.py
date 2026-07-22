"""Tests for the Playwright driver. The field extractor runs against a local HTML
fixture (file://, no network). Skips cleanly if playwright isn't installed.
"""
import pytest

pytest.importorskip("playwright.sync_api")

from src import apply_driver

FORM_HTML = """<!doctype html><html><body>
<form>
  <div><label for="n">Full name *</label><input id="n" type="text" required></div>
  <div><label for="e">Email</label><input id="e" type="email"></div>
  <div><label for="r">Resume/CV</label><input id="r" type="file"></div>
  <div><label for="g">Gender</label>
    <select id="g">
      <option>Select ...</option><option>Male</option>
      <option>Female</option><option>Decline to self-identify</option>
    </select>
  </div>
  <div><label for="w">Why do you want to work here?</label><textarea id="w"></textarea></div>
  <div><label>Are you authorized to work?</label>
    <div class="select-shell"><div class="select__control">Select...</div>
      <input class="select__input" type="text"></div>
  </div>
  <input type="hidden" name="csrf" value="x">
</form>
</body></html>"""


def test_extract_fields_from_local_form(tmp_path):
    from playwright.sync_api import sync_playwright
    f = tmp_path / "form.html"
    f.write_text(FORM_HTML)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        page = b.new_page()
        page.goto(f.as_uri())
        fields = apply_driver.extract_fields(page)
        b.close()

    by_label = {x["label"]: x for x in fields}
    assert by_label["Full name"]["type"] == "text" and by_label["Full name"]["required"]
    assert by_label["Email"]["type"] == "email"
    assert by_label["Resume/CV"]["type"] == "file"           # file input detected
    assert by_label["Gender"]["tag"] == "select"
    assert "Male" in by_label["Gender"]["options"]           # select options captured
    assert by_label["Why do you want to work here?"]["type"] == "textarea"
    assert all(x["type"] != "hidden" for x in fields)        # hidden field excluded
    # react-select widget extracted with its label; its inner input NOT double-counted
    assert by_label["Are you authorized to work?"]["type"] == "select-widget"
    assert sum(1 for x in fields if x["type"] == "select-widget") == 1


def test_hold_for_review_maps_input_to_status(monkeypatch):
    row = {"company": "X", "title": "Y"}
    monkeypatch.setattr("builtins.input", lambda *_: "s")
    assert apply_driver._hold_for_review(row) == "submitted"
    monkeypatch.setattr("builtins.input", lambda *_: "")
    assert apply_driver._hold_for_review(row) == "reviewed"
    monkeypatch.setattr("builtins.input", lambda *_: "q")
    assert apply_driver._hold_for_review(row) == "__quit__"


def test_note_summarizes_decisions():
    decisions = [
        {"label": "Full name", "action": "fill"},
        {"label": "Why us", "action": "draft", "drafted": True},
        {"label": "Current company", "action": "needs_human", "reason": "unmapped"},
    ]
    n = apply_driver._note(decisions)
    assert "filled=1" in n
    assert "drafted=Why us" in n
    assert "needs_human=Current company" in n
    assert "NOT submitted" in n
