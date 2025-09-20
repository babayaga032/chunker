#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API: Upload file → Extract → Chunk → LLM classify content_type → Embed (1536-d) → Insert into Supabase
"""

import os, io, re, uuid, hashlib
import fitz  # PyMuPDF for PDF
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from openai import OpenAI
from supabase import create_client
import tiktoken
from docx import Document as DocxDocument
import pandas as pd

# -------------------------------------------------------------------
# Config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client   = OpenAI(api_key=OPENAI_KEY)

EMBED_MODEL = "text-embedding-3-small"  # 1536 dimensions
CHUNK_SIZE  = 800
CHUNK_OVERLAP = 120

# Candidate content types
CONTENT_TYPES = [
    "lesson","assessment","exam","assignment","homework",
    "practice","solutions","answers","notes","rubric","worksheet",
    "textbook","course_mapping","curriculum","course_outline","general"
]

# -------------------------------------------------------------------
# Helpers

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def normalize_text(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def pdf_to_text(b: bytes) -> str:
    doc = fitz.open("pdf", b)
    parts = [page.get_text("text") for page in doc]
    return "\n".join(parts)

def docx_to_text(b: bytes) -> str:
    with io.BytesIO(b) as f:
        doc = DocxDocument(f)
        return "\n".join([p.text for p in doc.paragraphs])

def plain_to_text(b: bytes) -> str:
    return b.decode("utf-8", errors="ignore")

def csv_xlsx_to_text(b: bytes, ext: str) -> str:
    with io.BytesIO(b) as f:
        if ext == ".csv":
            df = pd.read_csv(f, dtype=str).fillna("")
        else:
            df = pd.read_excel(f, dtype=str).fillna("")
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"]*len(df.columns)) + " |"
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)

def chunk_text(text: str) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    chunks, step = [], CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(toks), step):
        piece = toks[start:start+CHUNK_SIZE]
        chunks.append(enc.decode(piece))
    return [c.strip() for c in chunks if c.strip()]

def classify_content_type(text: str, filename: str) -> str:
    """
    Use LLM to classify file into one of the predefined CONTENT_TYPES.
    """
    prompt = f"""
Classify this educational file into one category only:
{", ".join(CONTENT_TYPES)}

File name: {filename}
Text sample:
{text[:800]}

Return only one word from the list above.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0
    )
    guess = resp.choices[0].message.content.strip().lower()
    return guess if guess in CONTENT_TYPES else "general"

# -------------------------------------------------------------------
# API

app = FastAPI()

@app.post("/ingest")
async def ingest(file: UploadFile, course_code: str = Form("MCV4U"), unit_number: int = Form(0)):
    try:
        # 1. Extract text
        data = await file.read()
        ext = os.path.splitext(file.filename)[1].lower()
        if ext == ".pdf":
            text = pdf_to_text(data)
        elif ext == ".docx":
            text = docx_to_text(data)
        elif ext in [".csv", ".xlsx"]:
            text = csv_xlsx_to_text(data, ext)
        else:
            text = plain_to_text(data)
        text = normalize_text(text)

        # 2. Chunk
        chunks = chunk_text(text)
        if not chunks:
            return JSONResponse({"status":"error","msg":"no text extracted"}, status_code=400)

        # 3. Classify content_type
        content_type = classify_content_type(text, file.filename)

        # 4. Embeddings
        embeddings = client.embeddings.create(
            model=EMBED_MODEL, input=chunks
        ).data

        # 5. Build rows
        rows = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            digest = sha1_hex(chunk)
            unique_id = f"{file.filename}__{i}__{digest[:8]}"
            metadata = {
                "file_name": file.filename,
                "course_code": course_code,
                "unit_number": unit_number,
                "content_type": content_type,
                "is_chunk": True,
                "chunk_number": i,
                "total_chunks": len(chunks),
            }
            rows.append({
                "unique_id": unique_id,
                "content": chunk,
                "embedding": "[" + ",".join(str(x) for x in emb.embedding) + "]",
                "metadata": metadata,
            })
        # 6. Insert into Supabase
        try:
            res = supabase.table("mcv4u_documents").upsert(
                rows, on_conflict="unique_id"
            ).execute()
            return {"status": "success", "inserted": len(rows), "content_type": content_type, "response": res.data}
        except Exception as db_err:
            import traceback
            return JSONResponse({
                "status": "error",
                "msg": str(db_err),
                "trace": traceback.format_exc()
            }, status_code=500)

    except Exception as e:
        return JSONResponse({"status":"error","msg":str(e)}, status_code=500)
