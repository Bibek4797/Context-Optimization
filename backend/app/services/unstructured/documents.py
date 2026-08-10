import streamlit as st
import uuid
import re
from app.services.unstructured import llm_client

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

def extract_text_from_file(uploaded_file) -> str:
    file_type = uploaded_file.name.split(".")[-1].lower()
    text = ""
    uploaded_file.seek(0)
    
    if file_type == "pdf":
        if pypdf is None and PyPDF2 is None:
            st.error("No PDF reading library is available. Please install 'pypdf' or 'PyPDF2' (pip install pypdf).")
            return ""
        text = ""
        # Try pypdf first (preferred)
        if pypdf is not None:
            try:
                reader = pypdf.PdfReader(uploaded_file)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
            except Exception as e:
                text = ""  # Will fall through to PyPDF2 if available

        # Fallback to PyPDF2 if pypdf failed or is None
        if not text and PyPDF2 is not None:
            try:
                uploaded_file.seek(0)
                reader = PyPDF2.PdfReader(uploaded_file)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
            except Exception as e2:
                st.error(f"Failed to read PDF with both pypdf and PyPDF2: {e2}")

    else:
        try:
            text = uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Failed to read text file: {e}")
            
    # Strip excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def resolve_coreferences(text: str) -> str:
    """Coreference Resolution Pre-Pass: Resolves ambiguous pronouns to explicit entity names."""
    if not text or len(text.strip()) == 0:
        return text
        
    # Attempt FastCoref neural resolution if available
    try:
        from fastcoref import FCoref
        model = FCoref(device="cpu")
        preds = model.predict(texts=[text[:10000]]) # Process first 10k chars for speed
        resolved = preds[0].get_resolved_text()
        if resolved and len(resolved) > 0:
            return resolved
    except Exception:
        pass

    # Lightweight Pattern-Based Coreference Fallback:
    # Resolves sentence-initial pronouns ('He also', 'She also', 'It is', 'They also')
    # by matching preceding capital entity nouns in adjacent sentences.
    try:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 1:
            return text
            
        resolved_sentences = []
        last_entity = ""
        
        for sent in sentences:
            # Detect primary noun entity in sentence (e.g. "Andrew Ng", "DeepLearning.AI")
            cap_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sent)
            if cap_matches:
                last_entity = cap_matches[0]
                
            # Replace ambiguous pronouns at start of sentence if previous entity exists
            if last_entity:
                sent = re.sub(r'^(He|She|It|They)\s+also\b', f'{last_entity} also', sent)
                sent = re.sub(r'^(He|She|It|They)\s+was\b', f'{last_entity} was', sent)
                sent = re.sub(r'^(He|She|It|They)\s+is\b', f'{last_entity} is', sent)
                
            resolved_sentences.append(sent)
            
        return " ".join(resolved_sentences)
    except Exception:
        return text

def summarize_document(text: str) -> str:
    if not text.strip():
        return "Empty text."
    
    words = text.split()
    word_count = len(words)
    preview = " ".join(words[:25]) + ("..." if word_count > 25 else "")
    return f"Local preview: {preview} ({word_count} words)"

def build_document_record(uploaded_file, text: str, summary: str) -> dict:
    file_type = uploaded_file.name.split(".")[-1].upper()
    size_kb = len(uploaded_file.getvalue()) / 1024.0
    char_count = len(text)
    
    return {
        "id": str(uuid.uuid4())[:8],
        "name": uploaded_file.name,
        "type": file_type,
        "size_kb": round(size_kb, 2),
        "char_count": char_count,
        "summary": summary,
        "text": text
    }

def process_uploaded_files(uploaded_files) -> list[dict]:
    docs = []
    for file in uploaded_files:
        text = extract_text_from_file(file)
        if not text:
            continue
        # Apply Coreference Resolution Pre-Pass to resolve pronouns
        text = resolve_coreferences(text)
        summary = ""
        if llm_client.is_configured():
            summary = summarize_document(text)
        doc_record = build_document_record(file, text, summary)
        docs.append(doc_record)
    return docs
