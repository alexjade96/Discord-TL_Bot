# Language code mapping: langdetect codes → mBART-50 src/tgt codes
LANGDETECT_TO_MBART: dict[str, str] = {
    "af": "af_ZA",
    "ar": "ar_AR",
    "az": "az_AZ",
    "bn": "bn_IN",
    "cs": "cs_CZ",
    "de": "de_DE",
    "en": "en_XX",
    "es": "es_XX",
    "et": "et_EE",
    "fa": "fa_IR",
    "fi": "fi_FI",
    "fr": "fr_XX",
    "gl": "gl_ES",
    "gu": "gu_IN",
    "he": "he_IL",
    "hi": "hi_IN",
    "hr": "hr_HR",
    "id": "id_ID",
    "it": "it_IT",
    "ja": "ja_XX",
    "ka": "ka_GE",
    "kk": "kk_KZ",
    "km": "km_KH",
    "ko": "ko_KR",
    "lt": "lt_LT",
    "lv": "lv_LV",
    "mk": "mk_MK",
    "ml": "ml_IN",
    "mn": "mn_MN",
    "mr": "mr_IN",
    "my": "my_MM",
    "ne": "ne_NP",
    "nl": "nl_XX",
    "pl": "pl_PL",
    "ps": "ps_AF",
    "pt": "pt_XX",
    "ro": "ro_RO",
    "ru": "ru_RU",
    "si": "si_LK",
    "sl": "sl_SI",
    "sv": "sv_SE",
    "sw": "sw_KE",
    "ta": "ta_IN",
    "te": "te_IN",
    "th": "th_TH",
    "tl": "tl_XX",
    "tr": "tr_TR",
    "uk": "uk_UA",
    "ur": "ur_PK",
    "vi": "vi_VN",
    "xh": "xh_ZA",
    "zh-cn": "zh_CN",
    "zh-tw": "zh_TW",
    "zu": "zu_ZA",
}

MBART_SUPPORTED: set[str] = set(LANGDETECT_TO_MBART.values())

LANGDETECT_TO_NAME: dict[str, str] = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "ga": "Irish",
    "gl": "Galician",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "xh": "Xhosa",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "zu": "Zulu",
}


def get_mbart_code(langdetect_code: str) -> str | None:
    """Return the mBART-50 language code for a given langdetect code, or None if unsupported."""
    return LANGDETECT_TO_MBART.get(langdetect_code.lower())


def get_language_name(langdetect_code: str) -> str:
    """Return the full language name for a langdetect code, e.g. 'pt' → 'Portuguese'."""
    return LANGDETECT_TO_NAME.get(langdetect_code.lower(), langdetect_code.upper())


# Maps user-supplied /lang hints (names, aliases, native names, codes) to
# langdetect-style codes used throughout the pipeline.
_HINT_ALIASES: dict[str, str] = {
    # Chinese
    "chinese": "zh-cn", "chinese simplified": "zh-cn", "simplified chinese": "zh-cn",
    "mandarin": "zh-cn", "putonghua": "zh-cn", "zh": "zh-cn", "zh-cn": "zh-cn",
    "chinese traditional": "zh-tw", "traditional chinese": "zh-tw",
    "cantonese": "zh-tw", "zh-tw": "zh-tw",
    # Japanese
    "japanese": "ja", "nihongo": "ja", "jp": "ja", "ja": "ja",
    # Korean
    "korean": "ko", "hangul": "ko", "kr": "ko", "ko": "ko",
    # Common European
    "russian": "ru", "ru": "ru",
    "spanish": "es", "espanol": "es", "es": "es",
    "french": "fr", "francais": "fr", "fr": "fr",
    "german": "de", "deutsch": "de", "de": "de",
    "portuguese": "pt", "pt": "pt",
    "italian": "it", "italiano": "it", "it": "it",
    "dutch": "nl", "nl": "nl",
    "polish": "pl", "pl": "pl",
    "ukrainian": "uk", "uk": "uk",
    "arabic": "ar", "ar": "ar",
    "turkish": "tr", "tr": "tr",
    "thai": "th", "th": "th",
    "vietnamese": "vi", "vi": "vi",
    "hindi": "hi", "hi": "hi",
    "indonesian": "id", "id": "id",
    "swedish": "sv", "sv": "sv",
    "danish": "da", "da": "da",
    "norwegian": "no", "no": "no",
    "finnish": "fi", "fi": "fi",
    "greek": "el", "el": "el",
    "hebrew": "he", "he": "he",
    "romanian": "ro", "ro": "ro",
    "czech": "cs", "cs": "cs",
    "hungarian": "hu", "hu": "hu",
    "english": "en", "en": "en",
}


def parse_language_hint(name: str) -> str | None:
    """Convert a user-supplied language name or code to a langdetect code.

    Returns None if the name is not recognised.

    Examples:
        'chinese'  -> 'zh-cn'
        'Japanese' -> 'ja'
        'ko'       -> 'ko'
        'gibberish' -> None
    """
    return _HINT_ALIASES.get(name.lower().strip())
