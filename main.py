#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized API: Upload file → Extract → Chunk → LLM classify → Embed → Insert into Supabase
Memory-efficient for large files (40-70MB) on Render free tier
"""
import os
import io, re, hashlib, json, logging, gc, tempfile
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from supabase import create_client
import tiktoken
from docx import Document as DocxDocument
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from contextlib import contextmanager
import asyncio

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
BATCH_SIZE = 50  # Process embeddings in batches
MAX_FILE_SIZE = 75 * 1024 * 1024  # 75MB limit

CONTENT_TYPES = [
    "lesson", "assessment", "exam", "assignment", "homework",
    "practice", "solutions", "answers", "notes", "rubric", "worksheet",
    "textbook", "course_mapping", "curriculum", "course_outline", "general"
]

# -------------------------------------------------------------------
# Memory management helpers
@contextmanager
def temp_file_cleanup(*temp_files):
    """Context manager to ensure temp files are cleaned up"""
    try:
        yield
    finally:
        for temp_file in temp_files:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logging.info(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logging.warning(f"Failed to clean up {temp_file}: {e}")

def force_gc():
    """Force garbage collection"""
    gc.collect()
    logging.info("Forced garbage collection")

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

def pdf_to_text_streaming(file_path: str) -> str:
    """Stream PDF text extraction to reduce memory usage"""
    try:
        doc = fitz.open(file_path)
        text_parts = []
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            text_parts.append(page_text)
            
            # Periodic cleanup for very large PDFs
            if page_num % 10 == 0:
                force_gc()
        
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        logging.error(f"PDF extraction failed: {e}")
        raise

def docx_to_text_streaming(file_path: str) -> str:
    """Stream DOCX text extraction"""
    try:
        doc = DocxDocument(file_path)
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        return "\n".join(text_parts)
    except Exception as e:
        logging.error(f"DOCX extraction failed: {e}")
        raise

def csv_xlsx_to_text_streaming(file_path: str, ext: str) -> str:
    """Stream CSV/Excel processing with chunking"""
    try:
        chunk_size = 1000  # Process 1000 rows at a time
        text_parts = []
        
        if ext == ".csv":
            for chunk_df in pd.read_csv(file_path, dtype=str, chunksize=chunk_size):
                chunk_df = chunk_df.fillna("")
                if len(text_parts) == 0:  # Add header only once
                    text_parts.append("| " + " | ".join(chunk_df.columns) + " |")
                    text_parts.append("| " + " | ".join(["---"] * len(chunk_df.columns)) + " |")
                
                for _, row in chunk_df.iterrows():
                    text_parts.append("| " + " | ".join(str(x) for x in row) + " |")
                
                force_gc()  # Cleanup after each chunk
        else:
            # For Excel, read in chunks if possible
            df = pd.read_excel(file_path, dtype=str).fillna("")
            text_parts.append("| " + " | ".join(df.columns) + " |")
            text_parts.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
            
            for _, row in df.iterrows():
                text_parts.append("| " + " | ".join(str(x) for x in row) + " |")
        
        return "\n".join(text_parts)
    except Exception as e:
        logging.error(f"CSV/Excel extraction failed: {e}")
        raise

def chunk_text_generator(text: str):
    """Generator to yield chunks without storing all in memory"""
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    step = CHUNK_SIZE - CHUNK_OVERLAP
    
    for start in range(0, len(toks), step):
        piece = toks[start:start + CHUNK_SIZE]
        chunk = enc.decode(piece).strip()
        if chunk:
            yield chunk

def analyze_batch_with_llm(batch_chunks: list, filename: str, folder_name: str = None, file_name: str = None) -> dict:
    """Use Gemini for metadata classification on batch text"""
    # Combine batch chunks for analysis (limit to avoid token limits)
    combined_text = "\n\n".join(batch_chunks[:3])  # Use first 3 chunks of batch
    sample_text = combined_text[:1200] if len(combined_text) > 1200 else combined_text
    
    prompt = f"""
    You are analyzing an educational document batch. Return a JSON object with exactly these fields:
    - "content_type": one of {CONTENT_TYPES}
    - "course_code": derive from file_name → return ALL CAPS
    - "unit_number": derive from file_name → integer only (0 if not found)
    - "keywords": array of 4-6 specific keywords based on this batch content

    File name: {file_name}
    Batch text sample: {sample_text}

    Respond ONLY with valid JSON, nothing else.
    """
    
    try:
        resp = gemini_chat_client.invoke(prompt)
        raw = resp.content.strip()
        logging.info(f"Batch analysis LLM response: {raw[:100]}...")

        # Clean JSON wrappers
        raw_clean = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.IGNORECASE)
        parsed = json.loads(raw_clean)
        
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            parsed = parsed[0]
            
    except Exception as e:
        logging.warning(f"Invalid JSON from LLM: {e}, falling back to defaults")
        parsed = {
            "content_type": "general",
            "course_code": folder_name.upper() if folder_name else "GENERAL",
            "unit_number": 0,
            "keywords": [],
        }

    # Validate and clean response
    parsed["content_type"] = parsed.get("content_type", "general").lower()
    if parsed["content_type"] not in CONTENT_TYPES:
        parsed["content_type"] = "general"
    
    parsed["course_code"] = parsed.get("course_code", "GENERAL").upper()
    
    try:
        parsed["unit_number"] = int(parsed.get("unit_number", 0))
    except (ValueError, TypeError):
        parsed["unit_number"] = 0
    
    if not isinstance(parsed.get("keywords"), list):
        parsed["keywords"] = []

    return parsed

def count_tokens(text: str) -> int:
    """Count tokens in text"""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

async def process_embeddings_in_batches(chunks: list, base_analysis: dict, filename: str, folder_name: str, file_name: str):
    """Process embeddings in batches with analysis per batch and yield results to save memory"""
    total_chunks = len(chunks)
    
    for batch_start in range(0, total_chunks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_chunks)
        batch_chunks = chunks[batch_start:batch_end]
        
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
        
        logging.info(f"Processing batch {batch_num}/{total_batches} - chunks {batch_start}-{batch_end-1}")
        
        # Count tokens in this batch
        batch_tokens = [count_tokens(chunk) for chunk in batch_chunks]
        avg_tokens = sum(batch_tokens) / len(batch_tokens)
        total_batch_tokens = sum(batch_tokens)
        
        logging.info(f"Batch {batch_num} token stats: avg={avg_tokens:.1f}, total={total_batch_tokens}, min={min(batch_tokens)}, max={max(batch_tokens)}")
        
        try:
            # Analyze this specific batch for better keywords
            logging.info(f"Running analysis on batch {batch_num}")
            batch_analysis = analyze_batch_with_llm(batch_chunks, filename, folder_name, file_name)
            
            # Merge with base analysis but keep batch-specific keywords
            analysis = {
                **base_analysis,  # Keep course_code, content_type, unit_number from first analysis
                "keywords": batch_analysis.get("keywords", []),  # Use batch-specific keywords
                "batch_number": batch_num,
                "batch_token_stats": {
                    "avg_tokens": round(avg_tokens, 1),
                    "total_tokens": total_batch_tokens,
                    "min_tokens": min(batch_tokens),
                    "max_tokens": max(batch_tokens)
                }
            }
            
            logging.info(f"Batch {batch_num} analysis: content_type={analysis['content_type']}, keywords={analysis['keywords']}")
            
            # Generate embeddings for this batch with retry logic
            embeddings = await generate_embeddings_with_retry(batch_chunks, batch_num)
            
            if any(len(e) != EMBED_DIM for e in embeddings):
                raise ValueError(f"Embedding dimension mismatch, expected {EMBED_DIM}")
            
            # Build rows for this batch
            batch_rows = []
            for i, (chunk, emb, token_count) in enumerate(zip(batch_chunks, embeddings, batch_tokens)):
                global_index = batch_start + i
                digest = sha1_hex(chunk)
                unique_id = f"{filename}__{global_index}__{digest[:8]}"
                
                metadata = {
                    **analysis,
                    "file_name": filename,
                    "folder_name": folder_name,
                    "original_file_name": file_name,
                    "is_chunk": True,
                    "chunk_number": global_index,
                    "total_chunks": total_chunks,
                    "chunk_tokens": token_count,
                }
                
                batch_rows.append({
                    "unique_id": unique_id,
                    "content": chunk,
                    "embedding": emb,
                    "metadata": metadata,
                })
            
            yield batch_rows
            
            # Cleanup
            del embeddings, batch_chunks, batch_rows, batch_analysis
            force_gc()
            
            # Small delay to prevent overwhelming the API
            await asyncio.sleep(0.2)
            
        except Exception as e:
            logging.error(f"Failed to process batch {batch_num} ({batch_start}-{batch_end}): {e}")
            raise

async def generate_embeddings_with_retry(batch_chunks: list, batch_num: int, max_retries: int = 3):
    """Generate embeddings with retry logic for network failures"""
    for attempt in range(max_retries):
        try:
            logging.info(f"Generating embeddings for batch {batch_num} (attempt {attempt + 1}/{max_retries})")
            embeddings = openai_embeddings.embed_documents(texts=batch_chunks)
            logging.info(f"Successfully generated {len(embeddings)} embeddings for batch {batch_num}")
            return embeddings
            
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to generate embeddings for batch {batch_num} after {max_retries} attempts: {e}")
                raise
            
            wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
            logging.warning(f"Embedding generation failed for batch {batch_num} (attempt {attempt + 1}): {e}")
            logging.info(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

# -------------------------------------------------------------------
# API
app = FastAPI()

@app.post("/ingest_with_meta")
async def ingest_with_meta(file: UploadFile, folder_name: str = Form(...), file_name: str = Form(...)):
    temp_file_path = None
    
    try:
        logging.info(f"Ingestion started for file: {file.filename} (folder={folder_name}, file_name={file_name})")
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Seek back to beginning
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_FILE_SIZE//1024//1024}MB")
        
        logging.info(f"File size: {file_size / 1024 / 1024:.2f} MB")
        
        # Save to temporary file to avoid keeping in memory
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file_path = temp_file.name
            
            # Stream file to disk in chunks
            while True:
                chunk = await file.read(8192)  # 8KB chunks
                if not chunk:
                    break
                temp_file.write(chunk)
        
        logging.info(f"File saved to temp location: {temp_file_path}")
        
        # Extract text based on file type
        ext = os.path.splitext(file.filename)[1].lower()
        
        with temp_file_cleanup(temp_file_path):
            if ext == ".pdf":
                text = pdf_to_text_streaming(temp_file_path)
            elif ext == ".docx":
                text = docx_to_text_streaming(temp_file_path)
            elif ext in [".csv", ".xlsx"]:
                text = csv_xlsx_to_text_streaming(temp_file_path, ext)
            else:
                with open(temp_file_path, 'rb') as f:
                    text = f.read().decode("utf-8", errors="ignore")
            
            text = normalize_text(text)
            
            if not text.strip():
                return JSONResponse({"status": "error", "msg": "no text extracted"}, status_code=400)
            
            logging.info(f"Extracted text length: {len(text)} characters")
            
            # Convert chunks generator to list (we need the count)
            chunks = list(chunk_text_generator(text))
            
            if not chunks:
                return JSONResponse({"status": "error", "msg": "no chunks generated"}, status_code=400)
            
            logging.info(f"Generated {len(chunks)} chunks")
            
            # Clear text from memory
            del text
            force_gc()
            
            # Initial metadata classification (for base info like course_code, content_type, unit)
            base_analysis = analyze_batch_with_llm(chunks[:3], file.filename, folder_name, file_name)
            logging.info(f"Base analysis complete: {base_analysis}")
            
            # Process embeddings in batches with per-batch analysis
            total_inserted = 0
            
            async for batch_rows in process_embeddings_in_batches(chunks, base_analysis, file.filename, folder_name, file_name):
                try:
                    res = supabase.table("mcv4u_documents").upsert(batch_rows, on_conflict="unique_id").execute()
                    total_inserted += len(batch_rows)
                    logging.info(f"Inserted batch: {len(batch_rows)} rows (total: {total_inserted})")
                except Exception as e:
                    logging.error(f"Database insert failed: {e}")
                    raise
            
            # Final cleanup
            del chunks
            force_gc()
            
            return {
                "status": "success", 
                "inserted": total_inserted, 
                "base_analysis": base_analysis,
                "file_size_mb": round(file_size / 1024 / 1024, 2),
                "total_chunks": total_inserted,
                "avg_chunk_tokens": "calculated_per_batch"
            }
    
    except Exception as e:
        logging.exception("Error in /ingest_with_meta")
        return JSONResponse({"status": "error", "msg": str(e)}, status_code=500)
    
    finally:
        # Ensure temp file cleanup even if something goes wrong
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logging.info("Cleaned up temp file in finally block")
            except:
                pass

@app.get("/health")
async def health_check():
    return {"status": "healthy", "max_file_size_mb": MAX_FILE_SIZE // 1024 // 1024}
