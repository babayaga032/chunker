#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API: Upload file → Extract → Chunk → LLM classify (content_type + course_code + unit_number + keywords) → Embed → Insert into Supabase
"""
import os
import io, re, hashlib, json, logging
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from supabase import create_client
import tiktoken
from docx import Document as DocxDocument
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------------------------------------------
# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------------------------------------------
# Config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Force OpenAI embeddings (1536 dimensions)
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

# Gemini only for metadata classification
gemini_chat_client = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY
)

EMBED_DIM = 1536
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

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
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)
    return s.strip()

def pdf_to_text(b: bytes) -> str:
    doc = fitz.open("pdf", b)
    return "\n".join(page.get_text("text") for page in doc)

def docx_to_text(b: bytes) -> str:
    with io.BytesIO(b) as f:
        doc = DocxDocument(f)
        return "\n".join(p.text for p in doc.paragraphs)

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
        piece = toks[start:start + CHUNK_SIZE]
        chunks.append(enc.decode(piece))
    return [c.strip() for c in chunks if c.strip()]

def analyze_document_with_llm(text: str, filename: str, folder_name: str = None, file_name: str = None) -> dict:
    """Use Gemini for metadata classification"""
    prompt = f"""
    You are analyzing an educational document. Return a JSON object with exactly these fields:
    - "content_type": one of {CONTENT_TYPES}
    - "course_code": derive from file_name → return ALL CAPS
    - "unit_number": derive from file_name → integer only (0 if not found)
    - "keywords": array of 3-4 short keywords

    File name: {file_name}
    Text sample (first 800 chars): {text[:800]}

    Respond ONLY with valid JSON, nothing else.
    """
    resp = gemini_chat_client.invoke(prompt)
    raw = resp.content.strip()
    logging.info(f"Raw LLM response: {raw}")

    try:
        # clean wrappers like ```json
        raw_clean = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.IGNORECASE)
        parsed = json.loads(raw_clean)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            parsed = parsed[0]
    except Exception:
        logging.warning("Invalid JSON from LLM, falling back to defaults")
        parsed = {
            "content_type": "general",
            "course_code": folder_name.upper() if folder_name else "GENERAL",
            "unit_number": 0,
            "keywords": [],
        }

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

@app.post("/ingest_with_meta")
async def ingest_with_meta(file: UploadFile, folder_name: str = Form(...), file_name: str = Form(...)):
    logging.info(f"Ingestion started for file: {file.filename} (folder={folder_name}, file_name={file_name})")
    try:
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

        # Chunk
        chunks = chunk_text(text)
        if not chunks:
            return JSONResponse({"status": "error", "msg": "no text extracted"}, status_code=400)

        # Metadata classification
        analysis = analyze_document_with_llm(text, file.filename, folder_name, file_name)

        # Embeddings (OpenAI only)
        embeddings = openai_embeddings.embed_documents(texts=chunks)
        if any(len(e) != EMBED_DIM for e in embeddings):
            raise ValueError(f"Embedding dimension mismatch, expected {EMBED_DIM}")

        logging.info(f"Generated {len(embeddings)} embeddings of {EMBED_DIM} dimensions each")

        # Build rows
        rows = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            digest = sha1_hex(chunk)
            unique_id = f"{file.filename}__{i}__{digest[:8]}"
            metadata = {
                **analysis,
                "file_name": file.filename,
                "folder_name": folder_name,
                "original_file_name": file_name,
                "is_chunk": True,
                "chunk_number": i,
                "total_chunks": len(chunks),
            }
            rows.append({
                "unique_id": unique_id,
                "content": chunk,
                "embedding": emb,
                "metadata": metadata,
            })

        # Insert into Supabase
        res = supabase.table("mcv4u_documents").upsert(rows, on_conflict="unique_id").execute()
        return {"status": "success", "inserted": len(rows), "analysis": analysis, "response": res.data}

    except Exception as e:
        logging.exception("Error in /ingest_with_meta")
        return JSONResponse({"status": "error", "msg": str(e)}, status_code=500)
