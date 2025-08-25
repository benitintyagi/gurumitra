# ingest/normalize_corpus.py
import os
import re
import json
import argparse
import pathlib
import unicodedata
from typing import Dict

# --- Windows-safe console ---
try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

LIGATURES = {
    "ﬂ": "fl", "ﬁ": "fi", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "᾿": "'", "ʹ": "'", "ˈ": "'", "’": "'", "“": '"', "”": '"',
}

# Characters in Private Use Area and other frequent junk ranges
JUNK_RANGES = [
    (0xE000, 0xF8FF),   # Private Use Area
    (0xF0000, 0xFFFFD), # PUA-A (rare)
    (0x100000, 0x10FFFD) # PUA-B (rare)
]

DIACRITIC_HINTS = set("āīūṛṝḷṅñṭḍṇśṣḥ ṃṁ")
DEVANAGARI_RANGE = (0x0900, 0x097F)

def in_ranges(ch: str, ranges) -> bool:
    cp = ord(ch)
    return any(a <= cp <= b for a, b in ranges)

def strip_non_text(s: str) -> str:
    # Remove control chars
    s = re.sub(r"[\x00-\x1F\x7F]", " ", s)
    # Remove Private Use / junk glyphs
    s = "".join("" if in_ranges(c, JUNK_RANGES) else c for c in s)
    # Collapse repeated symbols often seen in bad extracts
    s = re.sub(r"[•·◊◦■□▪▫◆◇◻◼◾◽∑§¥]+", " ", s)
    return s

def normalize_ligatures(s: str) -> str:
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    return s

def normalize_spaces(s: str) -> str:
    # Keep paragraph boundaries if present, then collapse inner whitespace
    s = re.sub(r"[ \t\r\f\v]+", " ", s)   # collapse spaces/tabs
    s = re.sub(r"\n{3,}", "\n\n", s)      # clamp giant newlines
    s = re.sub(r"[ ]?\n[ ]?", "\n", s)    # trim spaces around newlines
    return s.strip()

def strip_accents(s: str) -> str:
    # Convert to NFKD and drop combining marks
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def guess_lang(text: str) -> str:
    # Heuristic: if Devanagari present → "hi"; if Latin with Sanskrit diacritics → "sa-Latn"; else "en"
    if any(DEVANAGARI_RANGE[0] <= ord(c) <= DEVANAGARI_RANGE[1] for c in text):
        return "hi"
    low = text.lower()
    if any(ch in low for ch in DIACRITIC_HINTS):
        return "sa-Latn"
    return "en"

def clean_text(raw: str) -> str:
    t = strip_non_text(raw)
    t = normalize_ligatures(t)
    t = normalize_spaces(t)
    return t

def process_file(path_in: pathlib.Path, path_out: pathlib.Path, keep_diacritics: bool) -> Dict:
    with path_in.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    content_raw = doc.get("content", "")
    content_clean = clean_text(content_raw)

    # Language (override/augment meta)
    meta = dict(doc.get("meta", {}))
    meta["language"] = guess_lang(content_clean)

    # Dual text fields for downstream:
    text_raw = content_clean
    text_ascii = strip_accents(content_clean)

    out_doc = {
        "meta": meta,
        "content": text_raw if keep_diacritics else text_ascii,  # primary content field
        "text_raw": text_raw,     # preserved with diacritics
        "text_ascii": text_ascii  # stripped accents (search-friendly)
    }

    path_out.parent.mkdir(parents=True, exist_ok=True)
    with path_out.open("w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False)

    return {
        "in": str(path_in),
        "out": str(path_out),
        "chars_in": len(content_raw),
        "chars_out": len(out_doc["content"]),
        "lang": meta["language"]
    }

def main():
    ap = argparse.ArgumentParser(description="Normalize parsed PDF JSON into clean text for embeddings.")
    ap.add_argument("--in_dir", type=str, default="processed", help="Input directory containing *.json")
    ap.add_argument("--out_dir", type=str, default="processed_clean", help="Output directory")
    ap.add_argument("--keep-diacritics", action="store_true", help="Keep diacritics in main content field (also writes text_ascii)")
    args = ap.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir)

    if not in_dir.exists():
        raise SystemExit(f"Input directory not found: {in_dir}")

    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files in {in_dir}")

    print(f"Normalizing {len(files)} files...")
    total_chars_in = total_chars_out = 0

    for i, p in enumerate(files, 1):
        p_out = out_dir / p.name
        info = process_file(p, p_out, keep_diacritics=args.keep_diacritics)
        total_chars_in += info["chars_in"]
        total_chars_out += info["chars_out"]
        print(f"[{i}/{len(files)}] {p.name} → lang={info['lang']} chars={info['chars_out']}")

    print("Done.")
    print(f"Chars in  : {total_chars_in}")
    print(f"Chars out : {total_chars_out}")
    print(f"Output dir: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
