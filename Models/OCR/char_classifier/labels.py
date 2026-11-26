"""Class-label decoding shared by the dataset generator and every consumer.

Labels are the class folder names written by render_chars.py and
render_chars_context.py.  Anything that turns a prediction back into text must
decode it here, so a new label format cannot reach one consumer and not another.
"""

# Latin punctuation labels from render_chars.CHARSET_EXTENDED.
_PUNCT = {
    'exclamation': '!', 'period': '.', 'comma': ',', 'semicolon': ';',
    'colon': ':', 'question': '?', 'hyphen': '-', 'underscore': '_',
    'slash': '/', 'at': '@', 'hash': '#', 'percent': '%', 'ampersand': '&',
    'plus': '+', 'equals': '=', 'lparen': '(', 'rparen': ')',
    'lbracket': '[', 'rbracket': ']',
}

_PREFIX_SCRIPT = {
    'cap': 'latin', 'low': 'latin', 'dig': 'latin', 'punct': 'latin',
    'hira': 'kana', 'kata': 'kana', 'syl': 'hangul', 'cjk': 'cjk',
}


def label_to_char(label: str) -> str:
    """'cap_A' -> 'A', 'punct_at' -> '@', 'cjk_4e00' -> U+4E00.

    Returns the label unchanged when it is not a known class label.
    """
    prefix, sep, rest = label.partition('_')
    if not sep or prefix not in _PREFIX_SCRIPT:
        return label
    if prefix in ('cap', 'low', 'dig'):
        return rest
    if prefix == 'punct':
        return _PUNCT.get(rest, label)
    try:
        return chr(int(rest, 16))
    except (ValueError, OverflowError):
        return label


def label_script(label: str) -> str:
    """Script a label belongs to: 'latin' | 'kana' | 'hangul' | 'cjk' | 'unknown'."""
    return _PREFIX_SCRIPT.get(label.partition('_')[0], 'unknown')
