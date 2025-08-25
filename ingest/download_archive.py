# ingest/download_archive.py
import os, json, pathlib
from internetarchive import search_items, get_item

DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(exist_ok=True)

# Example query terms/collections — adjust to your sources:
QUERIES = [
    'creator:"Max Müller" AND mediatype:text',
    'title:"Rig Veda" AND mediatype:text',
    'subject:"Hinduism" AND mediatype:text'
]

def main():
    seen = set()
    for q in QUERIES:
        for res in search_items(q):
            identifier = res.get('identifier')
            if not identifier or identifier in seen: continue
            seen.add(identifier)
            item = get_item(identifier)
            # Prefer text-friendly formats: .txt, .pdf, .djvu, .xml
            files = [f for f in item.files if f["name"].lower().endswith((".pdf", ".txt", ".djvu", ".xml"))]
            out_dir = DATA_DIR / identifier; out_dir.mkdir(exist_ok=True)
            for f in files:
                print("Downloading", f["name"])
                item.download(files=[f["name"]], destdir=str(out_dir), ignore_existing=True)

if __name__ == "__main__":
    main()
