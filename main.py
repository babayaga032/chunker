#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API: Upload file → Extract → Chunk → LLM classify content_type + course_code + unit_number + keywords → Embed → Insert into Supabase
"""

import os, io, re, uuid, hashlib, json, logging
import fitz  # PyMuPDF for PDF
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from supabase import create_client
import tiktoken
from docx import Document as DocxDocument
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------------------------------------------
# Config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
gemini_client = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GEMINI_API_KEY
)
gemini_chat_client = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY
)

EMBED_MODEL = "models/embedding-001"  # 768 dims for Gemini
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Candidate content types
CONTENT_TYPES = [
    "lesson", "assessment", "exam", "assignment", "homework",
    "practice", "solutions", "answers", "notes", "rubric", "worksheet",
    "textbook", "course_mapping", "curriculum", "course_outline", "general"
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
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)

def chunk_text(text: str) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    chunks, step = [], CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(toks), step):
        piece = toks[start : start + CHUNK_SIZE]
        chunks.append(enc.decode(piece))
    return [c.strip() for c in chunks if c.strip()]

def analyze_document_with_llm(text: str, filename: str) -> dict:
    """
    Ask Gemini to classify content_type, detect course_code + unit_number,
    and generate keywords. Must return valid JSON.
    """
    prompt = f"""
You are analyzing an educational document. 
Return a JSON object with exactly these fields:

- "content_type": one of {CONTENT_TYPES}
- "course_code": always ALL CAPS string (return "GENERAL" if not found)
- "unit_number": integer only (0 if not found)
- "keywords": array of 3-4 short keywords

Filename: {filename}
Text sample (first 800 chars):
{text[:800]}

Respond ONLY with valid JSON, nothing else.
"""

    resp = gemini_chat_client.invoke(prompt)
    raw = resp.content.strip()
    logging.info(f"Raw LLM response: {raw}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logging.warning("LLM did not return valid JSON. Falling back to defaults.")
        parsed = {
            "content_type": "general",
            "course_code": "GENERAL",
            "unit_number": 0,
            "keywords": [],
        }

    # Normalize
    parsed["content_type"] = parsed.get("content_type", "general").lower()
    if parsed["content_type"] not in CONTENT_TYPES:
        parsed["content_type"] = "general"
    parsed["course_code"] = parsed.get("course_code", "GENERAL").upper()
    try:
        parsed["unit_number"] = int(parsed.get("unit_number", 0))
    except Exception:
        parsed["unit_number"] = 0
    if not isinstance(parsed.get("keywords"), list):
        parsed["keywords"] = []

    return parsed

# -------------------------------------------------------------------
# API

app = FastAPI()

@app.post("/ingest")
async def ingest(file: UploadFile):
    logging.info(f"Ingestion started for file: {file.filename}")
    try:
        # 1. Extract text
        data = await file.read()
        ext = os.path.splitext(file.filename)[1].lower()
        logging.info(f"Extracting text from {file.filename} with extension {ext}")
        if ext == ".pdf":
            text = pdf_to_text(data)
        elif ext == ".docx":
            text = docx_to_text(data)
        elif ext in [".csv", ".xlsx"]:
            text = csv_xlsx_to_text(data, ext)
        else:
            text = plain_to_text(data)
        text = normalize_text(text)
        logging.info(f"Text extracted and normalized. Length: {len(text)}")

        # 2. Chunk
        chunks = chunk_text(text)
        if not chunks:
            logging.warning(f"No text extracted for file: {file.filename}")
            return JSONResponse(
                {"status": "error", "msg": "no text extracted"}, status_code=400
            )
        logging.info(f"Text chunked into {len(chunks)} chunks.")

        # 3. LLM analysis (content_type + course_code + unit_number + keywords)
        analysis = analyze_document_with_llm(text, file.filename)
        content_type = analysis["content_type"]
        course_code = analysis["course_code"]
        unit_number = analysis["unit_number"]
        keywords = analysis["keywords"]
        logging.info(
            f"LLM analysis: type={content_type}, course={course_code}, unit={unit_number}, keywords={keywords}"
        )

        # 4. Embeddings
        emb_resp = gemini_client.embed_documents(texts=chunks)
        embeddings = emb_resp
        logging.info(f"Embeddings generated for {len(embeddings)} chunks.")

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
                "keywords": keywords,
                "is_chunk": True,
                "chunk_number": i,
                "total_chunks": len(chunks),
            }
            rows.append(
                {
                    "unique_id": unique_id,
                    "content": chunk,
                    "embedding": emb,
                    "metadata": metadata,
                }
            )
        logging.info(f"Prepared {len(rows)} rows for database insertion.")

        # 6. Insert into Supabase
        try:
            res = supabase.table("mcv4u_documents").upsert(
                rows, on_conflict="unique_id"
            ).execute()
            logging.info(f"Successfully inserted {len(rows)} rows into Supabase.")
            return {
                "status": "success",
                "inserted": len(rows),
                "analysis": analysis,
                "response": res.data,
            }
        except Exception as db_err:
            logging.error(
                f"Database insertion error for file {file.filename}: {db_err}",
                exc_info=True,
            )
            import traceback

            return JSONResponse(
                {"status": "error", "msg": str(db_err), "trace": traceback.format_exc()},
                status_code=500,
            )

    except Exception as e:
        logging.error(
            f"An unexpected error occurred during ingestion for file {file.filename}: {e}",
            exc_info=True,
        )
        return JSONResponse({"status": "error", "msg": str(e)}, status_code=500)
