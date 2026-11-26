"""Guards against the generator, label decoder and evaluator drifting apart --
pure data checks, no model/GPU/network involved."""

import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "utils", "generators"))

import pytest
import render_chars
import render_chars_context
from char_classifier import eval_real
from char_classifier.labels import label_script, label_to_char
from char_classifier.vocab import failures, vocab_reports

CHARSETS = {
    "latin": render_chars.CHARSET_EXTENDED,
    "kana": render_chars._build_kana(),
    "hangul": render_chars._build_hangul(500),
    "cjk": render_chars._build_cjk(3000),
}


@pytest.mark.parametrize("script", sorted(CHARSETS))
def test_every_generated_label_decodes_to_its_character(script):
    for label, char in CHARSETS[script]:
        assert label_to_char(label) == char
        assert label_script(label) == script


def test_unknown_label_is_returned_unchanged():
    assert label_to_char("?") == "?"
    assert label_to_char("mystery_41") == "mystery_41"


def test_eval_crop_scale_matches_generator():
    midpoint = (render_chars_context.MIN_CROP_SCALE
                + render_chars_context.MAX_CROP_SCALE) / 2
    assert eval_real.GLYPH_SCALE == pytest.approx(midpoint)
    assert eval_real.TILE_SIZE == render_chars_context.TILE_SIZE


def test_vocab_report_flags_narrow_and_out_of_script_lists():
    reports = vocab_reports(["cjk_4e00", "cjk_304b"])
    messages = " ".join(failures(reports, 0.9))
    assert "outside the script" in messages
    assert "coverage" in messages


def test_complete_latin_list_has_no_problems():
    labels = [label for label, _ in render_chars.CHARSET_ALPHA]
    assert failures(vocab_reports(labels), 1.0) == []
