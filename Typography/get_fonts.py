import os
import shutil
import unicodedata
import argparse
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from fontTools.ttLib import TTFont

# ============================================================
# CONFIGURATION
# ============================================================

SYSTEM_FONTS_DIR = r"C:\Windows\Fonts"
WORKING_FONTS_DIR = "windows-fonts"
DATASET_DIR = "font-dataset"

FONT_SIZE = 64
TILE_SIZE = 128
PAGE_WIDTH = 2048
PAGE_HEIGHT = 1024
MARGIN = 20
LINE_SPACING = 20

PREVIEW_WIDTH = 1600
PREVIEW_HEIGHT = 600
PREVIEW_BG = "white"
PREVIEW_TEXT_COLOR = "black"

# Sizes rendered for each text corpus page (pt)
TEXT_PAGE_SIZES = [20, 32, 48]

# Additional tile-grid render sizes beyond the default FONT_SIZE=64
# Produces {block}_s{size}_{page}_{family}-{style}.png alongside the default tiles
EXTRA_TILE_SIZES = [32, 96]

# Script-specific running-text samples for non-Latin Unicode blocks.
# Each entry is keyed by the matching UNICODE_BLOCKS name; skipped if the font
# has no cmap coverage for that block.
SCRIPT_PANGRAMS = {
    "cyrillic": [
        "Съешь же ещё этих мягких французских булок, да выпей же чаю.",
        "Широкая электрификация южных губерний даст мощный толчок подъёму.",
        "В чащах юга жил бы цитрус? Да, но фальшивый экземпляр!",
    ],
    "greek": [
        "Θέλει αρετή και τόλμη η ελευθερία. (Ανδρέας Κάλβος)",
        "Ξεσκεπάζω την ψυχοφθόρα βδελυγμία.",
        "Αβγδεζηθικλμνξοπρστυφχψω  ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
    ],
    "arabic": [
        "نص حكيم له سر قاطع وذو شأن عظيم مكتوب على ثوب أبيض.",
        "أبجد هوز حطي كلمن سعفص قرشت ثخذ ضظغ",
    ],
    "hebrew": [
        "דג סקרן שט בים מאוכזב ולפתע מצא לו חברה.",
        "אבגדהוזחטיכלמנסעפצקרשת",
    ],
    "devanagari": [
        "ऋषियों को सताने वाले दुष्ट राक्षसों के राजा रावण का सर्वनाश।",
        "अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधन",
    ],
    "thai": [
        "เป็นมนุษย์สุดประเสริฐเลิศคุณค่า กว่าบรรดาฝูงสัตว์เดรัจฉาน",
        "กขคงจฉชซญดตถทธนบปผฝพฟภมยรลวศษสหอฮ",
    ],
    "georgian": [
        "გთხოვთ ახლავე გაიაროთ რეგისტრაცია Unicode-ის კონფერენციაზე.",
        "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ",
    ],
    "armenian": [
        "Բոլոր մարդիկ ծնվում են ազատ ու հավասար իրենց արժանապատվությամբ։",
        "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖ",
    ],
}

# Running-text corpora: each value is a list of lines to render on one page
TEXT_CORPORA = {
    "pangrams": [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "Sphinx of black quartz, judge my vow.",
        "How vexingly quick daft zebras jump!",
        "Jackdaws love my big sphinx of quartz.",
        "The five boxing wizards jump quickly.",
        "Waltz, bad nymph, for quick jigs vex!",
        "Bright vixens jump; dozy fowl quack.",
        "Blowzy night-frumps vex'd Jack Q.",
        "Cwm fjord bank glyphs vext quiz.",
    ],
    "lorem": [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit,",
        "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco",
        "laboris nisi ut aliquip ex ea commodo consequat. Duis aute",
        "irure dolor in reprehenderit in voluptate velit esse cillum",
        "dolore eu fugiat nulla pariatur. Excepteur sint occaecat",
        "cupidatat non proident, sunt in culpa qui officia deserunt",
        "mollit anim id est laborum. Sed ut perspiciatis unde omnis",
        "iste natus error sit voluptatem accusantium doloremque.",
    ],
    "numerals": [
        "0  1  2  3  4  5  6  7  8  9",
        "1,234,567.89     $9.99     €1.50     £2.75     ¥300",
        "100% + 50% = 150%     −273.15 °C     +3.14",
        "3.14159265   2.71828182   1.61803399   0.57721566",
        "Phone: (555) 867-5309     ZIP: 10001-0001",
        "2024-01-01   12:30:59   UTC−05:00",
        "#1 Best Seller — Vol. III, No. 7",
        "Section 4.2.1: see pp. 99–104 & App. B",
    ],
    "mixed_case": [
        "ABCDEFGHIJKLM   NOPQRSTUVWXYZ",
        "abcdefghijklm   nopqrstuvwxyz",
        "Aa Bb Cc Dd Ee Ff Gg Hh Ii Jj Kk Ll Mm",
        "Nn Oo Pp Qq Rr Ss Tt Uu Vv Ww Xx Yy Zz",
        "Hello, World!   Welcome to Typography.",
        "CamelCase   PascalCase   snake_case   kebab-case",
        "Title Case   SCREAMING SNAKE   ALL CAPS   lower",
        "The QUICK Brown FOX Jumps OVER The LAZY Dog.",
    ],
    "punctuation": [
        ". , ; : ! ? ' \" ( ) [ ] { } < > / \\",
        "— – - ... · • † ‡ § ¶ @ # % ^ & * ~ ` |",
        "© ® ™ ° ± × ÷ = ≠ ≈ ∞ √ ∑ ∏ Δ",
        "\"Quoted text,\" she said. 'Indeed!' he replied.",
        "It’s a contraction—it’s a test. (See footnote¹.)",
        "Price: $1,299.00*   *Taxes & fees not included.",
        "Hello—world! [Brackets], {Braces}, <Angles>.",
    ],
}

UNICODE_BLOCKS = {
    "latin": (0x0000, 0x024F),
    "greek": (0x0370, 0x03FF),
    "cyrillic": (0x0400, 0x04FF),
    "armenian": (0x0530, 0x058F),
    "hebrew": (0x0590, 0x05FF),
    "arabic": (0x0600, 0x06FF),
    "devanagari": (0x0900, 0x097F),
    "thai": (0x0E00, 0x0E7F),
    "georgian": (0x10A0, 0x10FF),
    "ethiopic": (0x1200, 0x137F),
    "symbols": (0x2000, 0x206F),
    "cjk": (0x4E00, 0x9FFF),
}

EXCLUDED_RANGES = [
    (0x1F000, 0x1FAFF),
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF),
    (0xD800, 0xDFFF),
]


# ============================================================
# HELPERS
# ============================================================

def is_excluded(cp):
    return any(start <= cp <= end for start, end in EXCLUDED_RANGES)


def is_printable(cp):
    try:
        ch = chr(cp)
        if unicodedata.category(ch).startswith("C"):
            return False
        if is_excluded(cp):
            return False
        return True
    except:
        return False


def normalize_name(name):
    name = name.replace("_", " ").replace("-", " ")
    name = " ".join(part for part in name.split() if part)
    return name.title().replace(" ", "-")


# ============================================================
# STYLE DETECTION
# ============================================================

def detect_style(weight, italic):
    if weight is None:
        # Fallback if OS/2 missing
        return "Italic" if italic else "Regular"
    if weight >= 600 and italic:
        return "BoldItalic"
    if weight >= 600:
        return "Bold"
    if italic:
        return "Italic"
    return "Regular"


# ============================================================
# METADATA + CMAP EXTRACTION
# ============================================================

def extract_font_metadata_and_cmap(font_path):
    meta = {
        "family": None,
        "weight": None,
        "italic": False,
        "serif": None,
        "monospace": None,
    }
    cmap_codepoints = set()

    try:
        tt = TTFont(font_path)

        # Name table → family name
        name_table = tt["name"]
        for record in name_table.names:
            if record.nameID == 1:
                meta["family"] = record.toUnicode()

        # OS/2 table
        if "OS/2" in tt:
            os2 = tt["OS/2"]
            meta["weight"] = getattr(os2, "usWeightClass", None)
            meta["italic"] = bool(getattr(os2, "fsSelection", 0) & 1)

            serif_type = getattr(os2, "panose", None)
            if serif_type is not None:
                serif_val = serif_type.bSerifStyle
                meta["serif"] = serif_val not in (0, 1, 2, 11)

        # POST table
        if "post" in tt:
            post = tt["post"]
            meta["monospace"] = bool(getattr(post, "isFixedPitch", 0))

        # CMAP table → supported codepoints
        if "cmap" in tt:
            for table in tt["cmap"].tables:
                cmap_codepoints.update(table.cmap.keys())

        tt.close()

    except Exception:
        pass

    return meta, cmap_codepoints


# ============================================================
# CMAP-BASED SUPPORT CHECKS
# ============================================================

def font_supports_char_cmap(cmap_set, ch):
    return ord(ch) in cmap_set


def font_supports_block_cmap(cmap_set, start, end):
    for cp in range(start, end + 1):
        if is_printable(cp) and cp in cmap_set:
            return True
    return False


# ============================================================
# COPY SYSTEM FONTS
# ============================================================

def copy_system_fonts():
    os.makedirs(WORKING_FONTS_DIR, exist_ok=True)

    files = [f for f in os.listdir(SYSTEM_FONTS_DIR) if f.lower().endswith((".ttf", ".otf"))]

    for file in tqdm(files, desc="Copying system fonts"):
        src = os.path.join(SYSTEM_FONTS_DIR, file)
        dst = os.path.join(WORKING_FONTS_DIR, file)
        try:
            shutil.copy2(src, dst)
        except:
            pass

    return WORKING_FONTS_DIR


# ============================================================
# NATURAL-SIZE TILE RENDERING (NO SCALING)
# ============================================================

def render_glyph_tile(font, ch, tile_size=TILE_SIZE):
    img = Image.new("L", (tile_size * 2, tile_size * 2), 0)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, font=font, fill=255)

    bbox = img.getbbox()
    if not bbox:
        return None

    glyph = img.crop(bbox)
    gw, gh = glyph.size

    tile = Image.new("L", (tile_size, tile_size), 0)
    x = (tile_size - gw) // 2
    y = (tile_size - gh) // 2
    tile.paste(glyph, (x, y))

    return tile


# ============================================================
# DEFAULT PREVIEW IMAGE
# ============================================================

def render_preview_image(font, output_dir, family, style):
    preview = Image.new("RGB", (PREVIEW_WIDTH, PREVIEW_HEIGHT), PREVIEW_BG)
    draw = ImageDraw.Draw(preview)

    lines = [
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "abcdefghijklmnopqrstuvwxyz",
        "0123456789",
        "The quick brown fox jumps over the lazy dog.",
        "!@#$%^&*()[]{}<>?/\\|.,;:-_+=~`"
    ]

    y = 40
    for line in lines:
        draw.text((40, y), line, font=font, fill=PREVIEW_TEXT_COLOR)
        y += FONT_SIZE + 20

    fname = f"{family}-{style}_preview.png"
    preview.save(os.path.join(output_dir, fname))


# ============================================================
# TEXT PAGE RENDERING (RUNNING TEXT, MULTIPLE SIZES)
# ============================================================

def _filter_to_cmap(text, cmap_set):
    return "".join(ch if ord(ch) in cmap_set else " " for ch in text)


def render_text_page(font_path, corpus_name, lines, size, cmap_set, output_dir, family, style, dark=False):
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return

    bg  = "#1e1e1e" if dark else "white"
    fg  = "white"   if dark else "black"
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), bg)
    draw = ImageDraw.Draw(img)

    # Measure actual line height from a reference glyph
    bbox = font.getbbox("Ag")
    line_height = (bbox[3] - bbox[1]) + max(6, size // 4)

    y = MARGIN
    rendered_chars = 0

    for line in lines:
        if y + line_height >= PAGE_HEIGHT - MARGIN:
            break
        filtered = _filter_to_cmap(line, cmap_set)
        if filtered.strip():
            draw.text((MARGIN, y), filtered, font=font, fill=fg)
            rendered_chars += len(filtered.strip())
        y += line_height

    if rendered_chars < 10:
        return  # Skip pages with negligible content (e.g. CJK-only fonts on Latin corpus)

    mode  = "dark" if dark else "light"
    fname = f"text_{corpus_name}_{size:02}_{mode}_{family}-{style}.png"
    img.save(os.path.join(output_dir, fname))


def render_text_pages(font_path, cmap_set, output_dir, family, style):
    for corpus_name, lines in TEXT_CORPORA.items():
        for size in TEXT_PAGE_SIZES:
            for dark in (False, True):
                render_text_page(font_path, corpus_name, lines, size, cmap_set, output_dir, family, style, dark=dark)


def render_script_pages(font_path, cmap_set, output_dir, family, style):
    for block_name, lines in SCRIPT_PANGRAMS.items():
        if block_name not in UNICODE_BLOCKS:
            continue
        start, end = UNICODE_BLOCKS[block_name]
        if not font_supports_block_cmap(cmap_set, start, end):
            continue
        for size in TEXT_PAGE_SIZES:
            for dark in (False, True):
                render_text_page(font_path, block_name, lines, size, cmap_set, output_dir, family, style, dark=dark)


# ============================================================
# BLOCK RENDERING (CMAP-BASED SUPPORT)
# ============================================================

def render_block(font, chars, cmap_set, output_dir, block_name, family, style):
    page = 1
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")

    x, y = MARGIN, MARGIN
    glyphs_rendered = 0

    progress = tqdm(total=len(chars), leave=False)

    for ch in chars:
        progress.set_description(f"{family} | {style} | {block_name} | {page}")

        if not font_supports_char_cmap(cmap_set, ch):
            progress.update(1)
            continue

        tile = render_glyph_tile(font, ch)
        if tile is None:
            progress.update(1)
            continue

        glyphs_rendered += 1

        tile_rgb = Image.merge("RGB", (tile, tile, tile))
        tw, th = tile_rgb.size

        if x + tw + MARGIN >= PAGE_WIDTH:
            x = MARGIN
            y += th + LINE_SPACING

        if y + th + MARGIN >= PAGE_HEIGHT:
            fname = f"{block_name}_{page:03}_{family}-{style}.png"
            img.save(os.path.join(output_dir, fname))

            page += 1
            img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
            x, y = MARGIN, MARGIN

        img.paste(tile_rgb, (x, y))
        x += tw + 20

        progress.update(1)

    progress.close()

    if glyphs_rendered == 0:
        return

    fname = f"{block_name}_{page:03}_{family}-{style}.png"
    img.save(os.path.join(output_dir, fname))


# ============================================================
# FONT PROCESSING
# ============================================================

def render_font_images(font_path, raw_font_name, update=False, style_folders=False):
    font_meta, cmap_set = extract_font_metadata_and_cmap(font_path)

    family = font_meta["family"]
    if not family:
        family = normalize_name(raw_font_name)
    family = normalize_name(family)

    style = detect_style(font_meta["weight"], font_meta["italic"])

    if style_folders:
        output_dir = os.path.join(DATASET_DIR, family, style)
    else:
        output_dir = os.path.join(DATASET_DIR, family)

    os.makedirs(output_dir, exist_ok=True)

    try:
        font = ImageFont.truetype(font_path, FONT_SIZE)
    except:
        return

    render_preview_image(font, output_dir, family, style)
    render_text_pages(font_path, cmap_set, output_dir, family, style)
    render_script_pages(font_path, cmap_set, output_dir, family, style)

    for block_name, (start, end) in UNICODE_BLOCKS.items():

        # Fast skip: if cmap has no codepoints in this block, skip entirely
        if not font_supports_block_cmap(cmap_set, start, end):
            continue

        chars = [chr(cp) for cp in range(start, end + 1) if is_printable(cp)]

        # Default tile size
        render_block(font, chars, cmap_set, output_dir, block_name, family, style)

        # Extra tile sizes — filename uses block_name_s{size} prefix to avoid collision
        for tile_size in EXTRA_TILE_SIZES:
            try:
                font_at_size = ImageFont.truetype(font_path, tile_size)
            except Exception:
                continue
            render_block(font_at_size, chars, cmap_set, output_dir, f"{block_name}_s{tile_size}", family, style)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", "-u", action="store_true",
                        help="Re-render families even if they already exist")
    parser.add_argument("--style-folders", action="store_true",
                        help="Use Option A: family/style/ structure")
    args = parser.parse_args()

    os.makedirs(DATASET_DIR, exist_ok=True)

    fonts_root = copy_system_fonts()

    font_files = [
        os.path.join(fonts_root, f)
        for f in os.listdir(fonts_root)
        if f.lower().endswith((".ttf", ".otf"))
    ]

    for font_path in tqdm(font_files, desc="Processing font families"):
        raw_name = os.path.basename(font_path)
        render_font_images(font_path, raw_name, update=args.update, style_folders=args.style_folders)


if __name__ == "__main__":
    main()
