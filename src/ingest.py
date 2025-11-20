import os, json, uuid, argparse
import pdfplumber
from tqdm import tqdm

def chunk_text(text, size=800, overlap=100):
    words = text.split()
    i = 0
    chunks = []
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

def extract_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            pages.append((i+1, txt))
    return pages

def main(input_dir, out_file):
    out = []

    for file in tqdm(os.listdir(input_dir)):
        path = os.path.join(input_dir, file)

        # PDF
        if file.endswith(".pdf"):
            for page_num, text in extract_pdf(path):
                chunks = chunk_text(text)
                for idx, c in enumerate(chunks):
                    out.append({
                        "id": str(uuid.uuid4()),
                        "text": c,
                        "metadata": {"source": file, "page": page_num, "chunk": idx}
                    })

        # TXT / MD
        else:
            text = open(path, "r", encoding="utf-8").read()
            chunks = chunk_text(text)
            for idx, c in enumerate(chunks):
                out.append({
                    "id": str(uuid.uuid4()),
                    "text": c,
                    "metadata": {"source": file, "chunk": idx}
                })

    with open(out_file, "w") as f:
        for d in out:
            f.write(json.dumps(d) + "\n")

    print(f"Total chunks: {len(out)}")

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--input_dir", default="data/input_docs")
    a.add_argument("--out", default="data/docs.jsonl")
    args = a.parse_args()
    main(args.input_dir, args.out)
