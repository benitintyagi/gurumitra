# ingest/parse_pdf.py
import os, pathlib, json, re
from pypdf import PdfReader
from langdetect import detect
from bs4 import BeautifulSoup

DATA_DIR = pathlib.Path("data")
OUT_DIR = pathlib.Path("processed"); OUT_DIR.mkdir(exist_ok=True)

def extract_text_from_pdf(pdf_path: pathlib.Path) -> str:
    text_parts = []
    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        text = "\n".join(text_parts).strip()
        if text:
            return text
    except Exception:
        pass
    # OCR fallback for image-only PDFs (optional; omitted for brevity)
    # You can add pdf2image + pytesseract here if needed.
    return ""

def extract_text_from_txt(txt_path: pathlib.Path) -> str:
    return txt_path.read_text(encoding="utf-8", errors="ignore")

def extract_text_from_xml(xml_path: pathlib.Path) -> str:
    soup = BeautifulSoup(xml_path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    return soup.get_text(" ").strip()

def normalize_text(s: str) -> str:
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def process_item(item_dir: pathlib.Path):
    texts = []
    for p in item_dir.rglob("*"):
        low = p.name.lower()
        if low.endswith(".pdf"):
            t = extract_text_from_pdf(p)
        elif low.endswith(".txt"):
            t = extract_text_from_txt(p)
        elif low.endswith(".xml") or low.endswith(".djvu"):
            t = extract_text_from_xml(p)
        else:
            continue
        if t:
            texts.append(t)
    if not texts: return
    content = "\n\n".join(texts)
    lang = detect(content)
    meta = {"identifier": item_dir.name, "language": lang, "source": str(item_dir)}
    out = {"meta": meta, "content": normalize_text(content)}
    (OUT_DIR / f"{item_dir.name}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

def main():
    for item_dir in DATA_DIR.iterdir():
        if item_dir.is_dir():
            process_item(item_dir)

if __name__ == "__main__":
    main()
