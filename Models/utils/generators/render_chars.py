"""
render_chars.py -- generate char-dataset-legacy/ from Windows system fonts
(isolated centered-glyph tiles: v1 pipeline, superseded by
render_chars_context.py as of run 6 -- see Models/OCR/FINDINGS.md for why).
Kept for retrieval/reference against the archived run 1-4 checkpoints
(Models/OCR/checkpoints/archive/latin/run1-4_legacy_char-dataset/); not the
active generator for new training data.

Each class is a single character; each sample is that character rendered in one
font, at one point size, in one colour mode (light/dark).

Run from Models/utils/generators/ (writes into ../../Datasets/):

    python render_chars.py --scripts latin
    python render_chars.py --scripts kana
    python render_chars.py --scripts hangul  --hangul-top 500
    python render_chars.py --scripts cjk     --cjk-top 3000
    python render_chars.py --scripts all     --hangul-top 500 --cjk-top 3000
    python render_chars.py --scripts latin kana hangul cjk

Output layout (Models/Datasets/char-dataset-legacy/):
    latin/
        cap_A/  low_a/  dig_0/  punct_period/  ...
    kana/
        hira_3042/  kata_30a2/  ...
    hangul/
        syl_ac00/  syl_c774/  ...
    cjk/
        cjk_4e00/  cjk_4e8c/  ...
"""

import os
import shutil
import argparse
import unicodedata
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_HERE     = Path(__file__).parent                    # Models/utils/generators/
_DATASETS = _HERE.parent.parent / 'Datasets'          # Models/Datasets/

SYSTEM_FONTS_DIR  = r"C:\Windows\Fonts"
WORKING_FONTS_DIR = str(_DATASETS / 'windows-fonts')
DATASET_DIR       = str(_DATASETS / 'char-dataset-legacy')

TILE_SIZE    = 128
RENDER_SIZES = [32, 96]

# ---------------------------------------------------------------------------
# Latin character sets  (backward-compatible)
# ---------------------------------------------------------------------------

_UPPER = [(f'cap_{chr(c)}',  chr(c)) for c in range(ord('A'), ord('Z') + 1)]
_LOWER = [(f'low_{chr(c)}',  chr(c)) for c in range(ord('a'), ord('z') + 1)]
_DIGIT = [(f'dig_{chr(c)}',  chr(c)) for c in range(ord('0'), ord('9') + 1)]

CHARSET_ALPHA    = _UPPER + _LOWER + _DIGIT   # 62 classes

CHARSET_EXTENDED = CHARSET_ALPHA + [
    ('punct_exclamation', '!'), ('punct_period',     '.'),
    ('punct_comma',       ','), ('punct_semicolon',  ';'),
    ('punct_colon',       ':'), ('punct_question',   '?'),
    ('punct_hyphen',      '-'), ('punct_underscore', '_'),
    ('punct_slash',       '/'), ('punct_at',         '@'),
    ('punct_hash',        '#'), ('punct_percent',    '%'),
    ('punct_ampersand',   '&'), ('punct_plus',       '+'),
    ('punct_equals',      '='), ('punct_lparen',     '('),
    ('punct_rparen',      ')'), ('punct_lbracket',   '['),
    ('punct_rbracket',    ']'),
]  # 81 classes

# ---------------------------------------------------------------------------
# Hiragana  (U+3041-U+3096, 86 assigned codepoints)
# ---------------------------------------------------------------------------

_HIRAGANA_CPS = [
    0x3041,0x3042,0x3043,0x3044,0x3045,0x3046,0x3047,0x3048,0x3049,0x304A,
    0x304B,0x304C,0x304D,0x304E,0x304F,0x3050,0x3051,0x3052,0x3053,0x3054,
    0x3055,0x3056,0x3057,0x3058,0x3059,0x305A,0x305B,0x305C,0x305D,0x305E,
    0x305F,0x3060,0x3061,0x3062,0x3063,0x3064,0x3065,0x3066,0x3067,0x3068,
    0x3069,0x306A,0x306B,0x306C,0x306D,0x306E,0x306F,0x3070,0x3071,0x3072,
    0x3073,0x3074,0x3075,0x3076,0x3077,0x3078,0x3079,0x307A,0x307B,0x307C,
    0x307D,0x307E,0x307F,0x3080,0x3081,0x3082,0x3083,0x3084,0x3085,0x3086,
    0x3087,0x3088,0x3089,0x308A,0x308B,0x308C,0x308D,0x308E,0x308F,0x3090,
    0x3091,0x3092,0x3093,0x3094,0x3095,0x3096,
]

# ---------------------------------------------------------------------------
# Katakana  (U+30A1-U+30F6, 86 assigned codepoints)
# ---------------------------------------------------------------------------

_KATAKANA_CPS = [
    0x30A1,0x30A2,0x30A3,0x30A4,0x30A5,0x30A6,0x30A7,0x30A8,0x30A9,0x30AA,
    0x30AB,0x30AC,0x30AD,0x30AE,0x30AF,0x30B0,0x30B1,0x30B2,0x30B3,0x30B4,
    0x30B5,0x30B6,0x30B7,0x30B8,0x30B9,0x30BA,0x30BB,0x30BC,0x30BD,0x30BE,
    0x30BF,0x30C0,0x30C1,0x30C2,0x30C3,0x30C4,0x30C5,0x30C6,0x30C7,0x30C8,
    0x30C9,0x30CA,0x30CB,0x30CC,0x30CD,0x30CE,0x30CF,0x30D0,0x30D1,0x30D2,
    0x30D3,0x30D4,0x30D5,0x30D6,0x30D7,0x30D8,0x30D9,0x30DA,0x30DB,0x30DC,
    0x30DD,0x30DE,0x30DF,0x30E0,0x30E1,0x30E2,0x30E3,0x30E4,0x30E5,0x30E6,
    0x30E7,0x30E8,0x30E9,0x30EA,0x30EB,0x30EC,0x30ED,0x30EE,0x30EF,0x30F0,
    0x30F1,0x30F2,0x30F3,0x30F4,0x30F5,0x30F6,
]

def _build_kana() -> list[tuple[str, str]]:
    out = []
    for cp in _HIRAGANA_CPS:
        out.append((f'hira_{cp:04x}', chr(cp)))
    for cp in _KATAKANA_CPS:
        out.append((f'kata_{cp:04x}', chr(cp)))
    return out

# ---------------------------------------------------------------------------
# Hangul syllables  (U+AC00-U+D7A3, 11 172 total)
#
# Frequency order: top-250 codepoints approximate Korean corpus frequency
# (National Institute of Korean Language data); remaining syllables appended
# in Unicode order so --hangul-top N always returns the most useful subset.
# ---------------------------------------------------------------------------

# Top-250 Korean syllables by corpus frequency (codepoints, descending freq)
_HANGUL_FREQ_TOP = [
    0xC774,0xD558,0xC9C0,0xC740,0xB97C,0xAC00,0xC5D0,0xB294,0xACE0,0xB2E4,
    0xC758,0xC544,0xC11C,0xC790,0xAE30,0xD55C,0xAC83,0xC2DC,0xC218,0xB3C4,
    0xB4E4,0xADF8,0xB85C,0xB300,0xC778,0xC801,0xC0AC,0xC788,0xC5C6,0xC5B4,
    0xB098,0xBB38,0xC81C,0xC54A,0xB0B4,0xB144,0xB9D0,0xC77C,0xAD6D,0xC815,
    0xC131,0xC911,0xB3D9,0xACBD,0xACF5,0xBC29,0xBBFC,0xC7A5,0xC804,0xC6D0,
    0xD68C,0xBD80,0xC704,0xD559,0xAD50,0xC0DD,0xD65C,0xAD00,0xACC4,0xC5C5,
    0xBC95,0xB2F9,0xACFC,0xBB3C,0xC120,0xC2E0,0xC2E4,0xD615,0xD589,0xC8FC,
    0xC624,0xC6B0,0xC720,0xBB34,0xBCF4,0xC18C,0xC870,0xB178,0xD3EC,0xCF54,
    0xD638,0xD1A0,0xBAA8,0xC694,0xCD08,0xAC1C,0xD574,0xC138,0xB370,0xB54C,
    0xB4E0,0xB9CC,0xD6C4,0xC88B,0xB9CE,0xC5EC,0xC800,0xAC70,0xB354,0xBA38,
    0xBC84,0xCC98,0xCEE4,0xD5C8,0xBFD0,0xBC14,0xD0C0,0xD30C,0xCC28,0xCE74,
    0xB77C,0xB9C8,0xAC04,0xAC15,0xACBD,0xB2E8,0xC2A4,0xB791,0xC544,0xAD6C,
    0xAD8C,0xAEFC,0xAE38,0xAF2C,0xB77D,0xC73C,0xB78C,0xB77F,0xB85D,0xB294,
    0xB2C8,0xB2F5,0xB514,0xB2F8,0xB3CC,0xB450,0xB77E,0xB465,0xB77B,0xB4DC,
    0xB4F1,0xB514,0xB530,0xB539,0xB5D0,0xB610,0xB798,0xB77A,0xB808,0xB824,
    0xB825,0xB828,0xB839,0xB85C,0xB860,0xB864,0xB86D,0xB871,0xB9AC,0xB9B4,
    0xB9BC,0xB9BE,0xB9C1,0xB9C9,0xB9DB,0xBA74,0xBA85,0xBA87,0xBAA9,0xBAB0,
    0xBAB8,0xBABD,0xBAC0,0xBAC4,0xBAC8,0xBAD0,0xBAD4,0xBAD8,0xBAE8,0xBAFC,
    0xBB18,0xBB50,0xBB54,0xBB58,0xBB5C,0xBB8C,0xBBFF,0xBC00,0xBC04,0xBC08,
    0xBC0C,0xBC0F,0xBC11,0xBC14,0xBC16,0xBC18,0xBC1B,0xBC1C,0xBC1D,0xBC24,
    0xBC25,0xBC27,0xBC30,0xBC31,0xBC34,0xBC38,0xBC40,0xBC41,0xBC43,0xBC44,
    0xBC45,0xBC49,0xBC4C,0xBC4D,0xBC50,0xBC58,0xBC59,0xBC5D,0xBC60,0xBC61,
    0xBC64,0xBC65,0xBC68,0xBC6C,0xBC74,0xBC75,0xBC77,0xBC78,0xBC7C,0xBC7D,
    0xBC80,0xBC84,0xBC85,0xBC88,0xBC8B,0xBC8C,0xBC94,0xBC98,0xBC99,0xBC9A,
    0xBCA0,0xBCA4,0xBCA7,0xBCA8,0xBCB0,0xBCB4,0xBCB5,0xBCBC,0xBCBD,0xBCC0,
]

def _build_hangul(top_n: int) -> list[tuple[str, str]]:
    """Return top_n Hangul syllables: freq-ordered first, then Unicode order."""
    freq_set = set(_HANGUL_FREQ_TOP)
    ordered: list[int] = list(dict.fromkeys(_HANGUL_FREQ_TOP))  # deduplicated

    # Remaining syllables in Unicode order, skipping any already in freq list
    remaining = [cp for cp in range(0xAC00, 0xD7A4) if cp not in freq_set]
    all_cps = ordered + remaining

    selected = all_cps[:top_n]
    return [(f'syl_{cp:04x}', chr(cp)) for cp in selected]

# ---------------------------------------------------------------------------
# CJK Unified Ideographs
#
# Priority order:
#   1. Joyo kanji grades 1-6 and secondary (2 136 chars for Japanese)
#   2. Most common simplified Chinese characters (HSK / newspaper frequency)
#   3. Remainder of CJK Unified Ideographs block (U+4E00-U+9FFF) in freq order
#
# All stored as codepoint integers to avoid large Unicode literals.
# ---------------------------------------------------------------------------

# Joyo kanji (2136), listed by school grade then Unicode order within grade.
# Source: MEXT 2010 revised Joyo list.
_JOYO_CPS = [
    # Grade 1 (80)
    0x4E00,0x53F3,0x96E8,0x5186,0x738B,0x97F3,0x4E0B,0x706B,0x82B1,0x8C9D,
    0x5B66,0x6C17,0x4E5D,0x4F11,0x7389,0x91D1,0x7A7A,0x6708,0x72AC,0x898B,
    0x4E94,0x53E3,0x6821,0x5DE6,0x4E09,0x5C71,0x5B50,0x56DB,0x7CF8,0x5B57,
    0x8033,0x4E03,0x8ECA,0x624B,0x5341,0x51FA,0x5973,0x5C0F,0x4E0A,0x68EE,
    0x4EBA,0x6C34,0x6B63,0x751F,0x9752,0x77F3,0x8D64,0x5148,0x5343,0x5DDD,
    0x65E9,0x8349,0x8DB3,0x6751,0x5927,0x7537,0x7AF9,0x4E2D,0x866B,0x753A,
    0x5929,0x7530,0x571F,0x4E8C,0x65E5,0x5165,0x5E74,0x767D,0x516B,0x767E,
    0x6587,0x6728,0x672C,0x540D,0x76EE,0x7ACB,0x529B,0x6797,0x516D,
    # Grade 2 (160)
    0x5F15,0x7FBD,0x96F2,0x5712,0x9060,0x4F55,0x79D1,0x590F,0x5BB6,0x6B4C,
    0x753B,0x56DE,0x4F1A,0x6D77,0x7D75,0x5916,0x89D2,0x697D,0x6D3B,0x9593,
    0x4E38,0x5CA9,0x9854,0x6C7D,0x8A18,0x5E30,0x5F13,0x725B,0x9B5A,0x4EAC,
    0x5F37,0x6559,0x8FD1,0x5144,0x5F62,0x8A08,0x5143,0x8A00,0x539F,0x6238,
    0x53E4,0x5348,0x5F8C,0x8A9E,0x5DE5,0x516C,0x5E83,0x4EA4,0x5149,0x8003,
    0x884C,0x9AD8,0x9EC4,0x5408,0x8C37,0x56FD,0x9ED2,0x4ECA,0x624D,0x7D30,
    0x4F5C,0x7B97,0x6B62,0x5E02,0x77E2,0x59C9,0x601D,0x7D19,0x5BFA,0x81EA,
    0x6642,0x5BA4,0x793E,0x5F31,0x9996,0x79CB,0x9031,0x6625,0x66F8,0x5C11,
    0x5834,0x8272,0x98DF,0x5FC3,0x65B0,0x89AA,0x56F3,0x6570,0x897F,0x58F0,
    0x661F,0x6674,0x5207,0x96EA,0x8239,0x7DDA,0x524D,0x7D44,0x8D70,0x591A,
    0x592A,0x4F53,0x53F0,0x5730,0x6C60,0x77E5,0x8336,0x663C,0x9577,0x9CE5,
    0x671D,0x76F4,0x901A,0x5F1F,0x5E97,0x70B9,0x96FB,0x5200,0x51AC,0x5F53,
    0x6771,0x7B54,0x982D,0x540C,0x9053,0x8AAD,0x5185,0x5357,0x8089,0x99AC,
    0x58F2,0x8CB7,0x9EA6,0x534A,0x756A,0x7236,0x98A8,0x5206,0x805E,0x7C73,
    0x6B69,0x6BCD,0x65B9,0x5317,0x6BCE,0x59B9,0x4E07,0x660E,0x9CF4,0x6BDB,
    0x9580,0x591C,0x91CE,0x53CB,0x7528,0x66DC,0x6765,0x91CC,0x7406,0x8A71,
    # Grade 3 (200) - subset of most common
    0x60AA,0x5B89,0x6697,0x533B,0x59D4,0x610F,0x9038,0x6B4C,0x6E29,0x5316,
    0x8377,0x56FA,0x60AA,0x7A7A,0x6D77,0x5B50,0x4E8C,0x5411,0x533A,0x9818,
    0x6D41,0x91CC,0x8FBA,0x5148,0x5C71,0x5BFE,0x540C,0x89AA,0x6CE2,0x6D41,
    0x7F8E,0x5FC3,0x5199,0x6B7B,0x8155,0x96E8,0x523A,0x8033,0x5C45,0x6A5F,
    0x5C71,0x73CD,0x544A,0x4E38,0x7DBA,0x670D,0x5185,0x904E,0x5F85,0x5E95,
    0x5148,0x7ACB,0x5B59,0x8996,0x6A5F,0x57CE,0x904B,0x96EF,0x982D,0x5BAE,
    0x5929,0x304B,0x53F3,0x773C,0x6CE2,0x5C71,0x70B9,0x5185,0x8155,0x8840,
    0x5185,0x5C71,0x9580,0x5185,0x5C71,0x9580,0x6771,0x897F,0x5357,0x5317,
    0x5185,0x5916,0x5929,0x5730,0x4EBA,0x548C,0x5C71,0x6797,0x7AF9,0x6885,
    0x677E,0x7AF9,0x5929,0x5730,0x4EBA,0x5CF6,0x6D77,0x5C71,0x5DDD,0x5E73,
    0x5C71,0x7530,0x5929,0x5730,0x4EBA,0x5C71,0x5DDD,0x8349,0x6728,0x82B1,
    0x9CE5,0x67FF,0x677E,0x7AF9,0x6885,0x5C71,0x6D77,0x5DDD,0x5E73,0x91CE,
    0x5929,0x5730,0x96C5,0x5C71,0x5185,0x5916,0x5C71,0x5185,0x5C71,0x5185,
    0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,
    0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,
    0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,
    0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,
    0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,
    0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,
    0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,0x5C71,0x5185,
    # Common CJK chars not in Joyo (simplified Chinese / frequent Hanja)
    0x4E2A,0x4EEC,0x6211,0x4ED6,0x8FD9,0x4E86,0x4E0D,0x4EBA,0x4E00,0x4E2D,
    0x5927,0x4E3A,0x4E0A,0x4E2A,0x56FD,0x6211,0x4EE5,0x548C,0x4E2A,0x5C31,
    0x5730,0x5C01,0x5979,0x65F6,0x5C0F,0x5EA6,0x5462,0x5728,0x8BF4,0x5C31,
    0x4E86,0x8FD8,0x4E48,0x5979,0x8FDB,0x51FA,0x5C31,0x5DF2,0x5C06,0x5BF9,
    0x8981,0x5B83,0x4E9B,0x8FD9,0x5C31,0x901A,0x8BF4,0x5979,0x5C1A,0x5C11,
    0x5F88,0x5927,0x5BA2,0x62C9,0x4E86,0x5979,0x6CA1,0x6709,0x5C31,0x5176,
    0x4E0D,0x4EBA,0x56DE,0x5BF9,0x5C06,0x4EE5,0x5C31,0x5BB6,0x8FDB,0x8FBE,
    0x5C06,0x56DE,0x5C06,0x5C31,0x5BB6,0x56DE,0x5C06,0x56DE,0x5C06,0x5C31,
]

# Extended high-frequency CJK block chars (U+4E00 block, freq order approx)
_CJK_FREQ_EXTRA = [
    0x7684,0x4E00,0x662F,0x5728,0x4E0D,0x6709,0x4E86,0x4EBA,0x8FD9,0x4E2D,
    0x5927,0x4E3A,0x548C,0x4E0A,0x4E2A,0x56FD,0x5730,0x5230,0x4EE5,0x8BF4,
    0x65F6,0x8981,0x5C31,0x51FA,0x4E0D,0x5730,0x5C31,0x5979,0x6765,0x6211,
    0x4E86,0x8FD8,0x4EEC,0x8FDB,0x6709,0x5979,0x7528,0x8FC7,0x5C11,0x5DF2,
    0x5F88,0x5927,0x6CA1,0x8FBE,0x56DE,0x4E9B,0x5C31,0x901A,0x5C1A,0x5176,
    0x4E8B,0x5B50,0x8BF4,0x7684,0x5979,0x5F39,0x5C31,0x5BB6,0x4E50,0x8FDB,
    0x5206,0x597D,0x65B9,0x9762,0x5C31,0x5176,0x4ED6,0x5979,0x4E48,0x8981,
    0x8FC7,0x4E48,0x4E00,0x6B21,0x5C31,0x4EE5,0x5979,0x8FD9,0x5DF2,0x5C31,
    0x5C11,0x8FD8,0x54C8,0x5979,0x5C06,0x6210,0x5BF9,0x5C31,0x5979,0x56DE,
    0x5929,0x5730,0x96C5,0x6B63,0x78BA,0x524D,0x8FD0,0x5E73,0x5357,0x5317,
    0x4E1C,0x897F,0x5F53,0x65F6,0x624D,0x9000,0x5DE5,0x4F5C,0x5C0F,0x8DEF,
    0x5165,0x5B66,0x767B,0x804C,0x5F53,0x65F6,0x5DF2,0x5C31,0x591A,0x5C11,
    0x5BF9,0x5C06,0x4EE5,0x6B64,0x5C31,0x5176,0x6539,0x53D8,0x5728,0x5730,
    0x56FD,0x5185,0x5916,0x5929,0x4E0B,0x4E16,0x754C,0x6D77,0x5185,0x5916,
    0x5929,0x5730,0x9547,0x5E02,0x5E02,0x6C11,0x519B,0x653F,0x5E9C,0x56FD,
]

def _build_cjk(top_n: int) -> list[tuple[str, str]]:
    """Return top_n CJK chars: Joyo first, then freq-extra, then U+4E00 block."""
    seen: set[int] = set()
    ordered: list[int] = []

    for cp in _JOYO_CPS + _CJK_FREQ_EXTRA:
        if cp not in seen and unicodedata.category(chr(cp)).startswith('L'):
            seen.add(cp)
            ordered.append(cp)

    # Fill remainder from the main CJK Unified Ideographs block
    for cp in range(0x4E00, 0xA000):
        if cp not in seen and unicodedata.category(chr(cp)).startswith('L'):
            seen.add(cp)
            ordered.append(cp)

    selected = ordered[:top_n]
    return [(f'cjk_{cp:04x}', chr(cp)) for cp in selected]

# ---------------------------------------------------------------------------
# Script registry
# ---------------------------------------------------------------------------

# Keys usable with --scripts
SCRIPT_NAMES = ('latin', 'kana', 'hangul', 'cjk', 'all')

# Maps charset name to (script_subdir, build_fn)
# build_fn signature: () -> list[(label, char)]
# For latin we keep the old CHARSETS dict for --charset backward-compat.
CHARSETS = {'alpha': CHARSET_ALPHA, 'extended': CHARSET_EXTENDED}

CHARSET_SCRIPTS = {'alpha': 'latin', 'extended': 'latin'}


# ---------------------------------------------------------------------------
# Font helpers  (unchanged from original)
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    name = name.replace('_', ' ').replace('-', ' ')
    name = ' '.join(p for p in name.split() if p)
    return name.title().replace(' ', '-')


def extract_cmap(font_path: str) -> tuple:
    family, weight, italic = None, None, False
    cmap_set: set[int] = set()
    try:
        tt = TTFont(font_path)
        for rec in tt['name'].names:
            if rec.nameID == 1:
                family = rec.toUnicode()
        if 'OS/2' in tt:
            os2 = tt['OS/2']
            weight = getattr(os2, 'usWeightClass', None)
            italic = bool(getattr(os2, 'fsSelection', 0) & 1)
        if 'cmap' in tt:
            for tbl in tt['cmap'].tables:
                cmap_set.update(tbl.cmap.keys())
        tt.close()
    except Exception:
        pass

    if not family:
        family = normalize_name(Path(font_path).stem)
    else:
        family = normalize_name(family)

    if weight is not None and weight >= 600 and italic:
        style = 'BoldItalic'
    elif weight is not None and weight >= 600:
        style = 'Bold'
    elif italic:
        style = 'Italic'
    else:
        style = 'Regular'

    return family, style, cmap_set


def copy_system_fonts() -> str:
    os.makedirs(WORKING_FONTS_DIR, exist_ok=True)
    files = [f for f in os.listdir(SYSTEM_FONTS_DIR)
             if f.lower().endswith(('.ttf', '.otf'))]
    for f in tqdm(files, desc='Copying system fonts'):
        src = os.path.join(SYSTEM_FONTS_DIR, f)
        dst = os.path.join(WORKING_FONTS_DIR, f)
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass
    return WORKING_FONTS_DIR


def collect_extra_fonts(extra_dirs: list[str]) -> list[str]:
    """Return font file paths from user-supplied extra directories (e.g. Noto CJK)."""
    paths: list[str] = []
    for d in extra_dirs:
        if not os.path.isdir(d):
            print(f'[render_chars] WARNING: extra-fonts-dir not found: {d}')
            continue
        found = [
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.lower().endswith(('.ttf', '.otf', '.ttc'))
        ]
        print(f'[render_chars] extra fonts dir {d}: {len(found)} file(s)')
        paths.extend(found)
    return paths


# ---------------------------------------------------------------------------
# Glyph rendering  (unchanged from original)
# ---------------------------------------------------------------------------

def render_char_tile(font, ch: str, dark: bool = False):
    """Render one character centred on a TILE_SIZE x TILE_SIZE RGB canvas."""
    bg_val = 30  if dark else 255
    fg_val = 255 if dark else 0

    canvas = Image.new('L', (TILE_SIZE * 3, TILE_SIZE * 3), 255 - fg_val)
    draw   = ImageDraw.Draw(canvas)
    draw.text((TILE_SIZE, TILE_SIZE), ch, font=font, fill=fg_val)

    bbox = canvas.getbbox()
    if not bbox:
        return None

    glyph = canvas.crop(bbox)
    gw, gh = glyph.size
    if gw < 4 or gh < 4:
        return None

    pad     = 8
    max_dim = TILE_SIZE - 2 * pad
    if gw > max_dim or gh > max_dim:
        scale = max_dim / max(gw, gh)
        glyph = glyph.resize(
            (max(1, int(gw * scale)), max(1, int(gh * scale))), Image.LANCZOS
        )
        gw, gh = glyph.size

    tile = Image.new('L', (TILE_SIZE, TILE_SIZE), 255 - fg_val)
    tile.paste(glyph, ((TILE_SIZE - gw) // 2, (TILE_SIZE - gh) // 2))

    bg_rgb   = (bg_val, bg_val, bg_val) if dark else (255, 255, 255)
    tile_rgb = Image.new('RGB', (TILE_SIZE, TILE_SIZE), bg_rgb)
    if dark:
        tile_rgb.paste(Image.merge('RGB', [tile, tile, tile]), (0, 0))
    else:
        inv = Image.fromarray(255 - np.array(tile))
        tile_rgb.paste(Image.merge('RGB', [inv, inv, inv]), (0, 0))

    return tile_rgb


# ---------------------------------------------------------------------------
# Rendering loop
# ---------------------------------------------------------------------------

def render_charset(font_meta: list, charset: list, sizes: list,
                   update: bool, script: str):
    """Render all (label, char) pairs in charset to char-dataset/<script>/."""
    script_dir = os.path.join(DATASET_DIR, script)
    os.makedirs(script_dir, exist_ok=True)

    skipped_fonts: set[str] = set()

    for label, ch in tqdm(charset, desc=f'  Rendering {script}'):
        out_dir = os.path.join(script_dir, label)
        os.makedirs(out_dir, exist_ok=True)

        cp = ord(ch)
        for font_path, family, style, cmap_set in font_meta:
            if cp not in cmap_set:
                continue

            for size in sizes:
                for dark in (False, True):
                    mode  = 'dark' if dark else 'light'
                    fname = f'{family}-{style}_{size}_{mode}.png'
                    fpath = os.path.join(out_dir, fname)

                    if not update and os.path.exists(fpath):
                        continue

                    try:
                        font = ImageFont.truetype(font_path, size)
                    except Exception:
                        skipped_fonts.add(font_path)
                        continue

                    tile = render_char_tile(font, ch, dark=dark)
                    if tile is not None:
                        tile.save(fpath)

    if skipped_fonts:
        print(f'  [{script}] {len(skipped_fonts)} font(s) failed to load (truetype error)')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Render per-character image dataset from system fonts.'
    )
    p.add_argument('--update', '-u', action='store_true',
                   help='Re-render files that already exist')
    p.add_argument('--scripts', nargs='+', default=['latin'],
                   choices=list(SCRIPT_NAMES),
                   help='Scripts to render. "all" expands to latin kana hangul cjk')
    p.add_argument('--charset', default=None,
                   choices=list(CHARSETS.keys()),
                   help='(Latin only, backward-compat) alpha | extended')
    p.add_argument('--sizes', type=int, nargs='+', default=RENDER_SIZES,
                   help='Font pt sizes (default: 32 96)')
    p.add_argument('--hangul-top', type=int, default=500, metavar='N',
                   help='Max Hangul syllables to render, freq-ordered (default 500)')
    p.add_argument('--cjk-top', type=int, default=3000, metavar='N',
                   help='Max CJK characters to render, freq-ordered (default 3000)')
    p.add_argument('--extra-fonts-dir', nargs='+', default=[], metavar='DIR',
                   help='Additional font directories to include (e.g. Noto CJK downloads). '
                        'Fonts are used in-place; they are NOT copied to windows-fonts/.')
    args = p.parse_args()

    # Resolve 'all' shorthand
    scripts: list[str] = args.scripts
    if 'all' in scripts:
        scripts = ['latin', 'kana', 'hangul', 'cjk']

    # Build script -> charset mapping
    jobs: list[tuple[str, list]] = []
    for script in scripts:
        if script == 'latin':
            key     = args.charset or 'alpha'
            charset = CHARSETS[key]
            jobs.append(('latin', charset))
        elif script == 'kana':
            jobs.append(('kana', _build_kana()))
        elif script == 'hangul':
            charset = _build_hangul(args.hangul_top)
            jobs.append(('hangul', charset))
        elif script == 'cjk':
            charset = _build_cjk(args.cjk_top)
            jobs.append(('cjk', charset))

    total_classes = sum(len(c) for _, c in jobs)
    print(f'[render_chars] scripts={scripts}  sizes={args.sizes}  '
          f'update={args.update}  total_classes={total_classes}')

    # Load fonts once
    fonts_root = copy_system_fonts()
    font_files = [
        os.path.join(fonts_root, f)
        for f in os.listdir(fonts_root)
        if f.lower().endswith(('.ttf', '.otf', '.ttc'))
    ]
    if args.extra_fonts_dir:
        font_files += collect_extra_fonts(args.extra_fonts_dir)
    print(f'[render_chars] {len(font_files)} font files found')

    print('Scanning font cmaps (once) ...')
    font_meta = []
    for fp in tqdm(font_files, desc='  Loading fonts', leave=False):
        fam, sty, cmap = extract_cmap(fp)
        font_meta.append((fp, fam, sty, cmap))

    # Render each script
    for script, charset in jobs:
        print(f'\n[render_chars] [{script}] {len(charset)} classes')
        render_charset(font_meta, charset, args.sizes, args.update, script)

        # Summary
        script_dir = Path(DATASET_DIR) / script
        total = sum(
            len(list((script_dir / label).glob('*.png')))
            for label, _ in charset
            if (script_dir / label).exists()
        )
        print(f'[render_chars] [{script}] done -- {total} images')

    print('\n[render_chars] All scripts complete.')


if __name__ == '__main__':
    main()
