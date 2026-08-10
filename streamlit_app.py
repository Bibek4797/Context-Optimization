from __future__ import annotations

import os
import shutil
import sys
import re
import time
from pathlib import Path
from uuid import uuid4
import streamlit as st
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"

# Streamlit Cloud cache busting: delete all __pycache__ and stale compiled bytecode (.pyc)
for root, dirs, files in os.walk(str(BACKEND_DIR)):
    for d in list(dirs):
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
            dirs.remove(d)
    for f in files:
        if f.endswith(".pyc") or f.endswith(".pyo"):
            try:
                os.remove(os.path.join(root, f))
            except Exception:
                pass

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Safely clear local app module cache without touching Streamlit internal runner modules
for mod_name in list(sys.modules.keys()):
    if (mod_name.startswith("app.") or mod_name == "app" or mod_name.startswith("backend.")) and not mod_name.startswith("streamlit"):
        try:
            del sys.modules[mod_name]
        except Exception:
            pass

from app.models.schemas import GraphDocument, QueryRecord, RepoMetadata, TreeNode, RepoStatus
from app.services.analysis_pipeline import AnalysisPipeline
from app.services.chat_service import ChatService
from app.services.codegraph_service import CodeGraphService
from app.services.file_utils import clean_repo_name, safe_extract_zip
from app.services.graph_retrieval_service import GraphRetrievalService
from app.services.graphify_service import GraphifyService
from app.services.llm.gemini import GeminiProvider
from app.services.llm.openai_compatible import GroqProvider, OpenRouterProvider
from app.services.repo_service import RepoService
from app.services.storage import LocalStorage
from app.services.token_service import TokenService
from app.services.tree_sitter_service import TreeSitterService

# Custom unstructured service imports
from app.services.unstructured.documents import process_uploaded_files
from app.services.unstructured.graph_builder import build_graph_from_documents
from app.services.unstructured.communities import detect_communities, summarize_all_communities
from app.services.unstructured.visualization import graph_to_pyvis
from app.services.agent_harness import AgentHarness


# ── Code Fix Extraction Helper ──
def _extract_code_fixes(text: str) -> list[dict]:
    """Parse all <code_fix> XML blocks from an LLM answer and return structured dicts."""
    if not text:
        return []
    # Normalize escaped/literal newlines first to prevent regex match failures
    normalized = text.replace("\\\\n", "\n").replace("\\n", "\n")
    fixes = []
    pattern = re.compile(
        r"<code_fix>[\s\n\r]*"
        r"<filepath>(?P<filepath>.*?)</filepath>[\s\n\r]*"
        r"<original_code>(?P<original>.*?)</original_code>[\s\n\r]*"
        r"<replacement_code>(?P<replacement>.*?)</replacement_code>[\s\n\r]*"
        r"</code_fix>",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(normalized):
        fixes.append({
            "filepath": m.group("filepath").strip(),
            "original": m.group("original").strip(),
            "replacement": m.group("replacement").strip(),
        })
    return fixes


def _clean_xml_from_text(text: str) -> str:
    """Remove <code_fix> blocks and normalize newlines for clean user display."""
    if not text:
        return ""
    normalized = text.replace("\\\\n", "\n").replace("\\n", "\n")
    clean = re.sub(
        r"<code_fix>.*?</code_fix>",
        "",
        normalized,
        flags=re.IGNORECASE | re.DOTALL
    )
    return clean.strip()


# ── Asset Token Helpers ──
def _get_active_codebase_tokens(repo_id: str | None) -> int:
    if not repo_id:
        return 0
    try:
        repo_root = storage.repo_source_dir(repo_id)
        files = storage.load_files(repo_id)
        total = 0
        from app.services.file_utils import read_text_lossy
        for rf in files:
            p = repo_root / rf.path
            if p.exists():
                text = read_text_lossy(p)
                total += token_service.estimate_tokens(text)
        return total
    except Exception:
        return 0

def _get_active_pdf_tokens() -> int:
    docs = st.session_state.get("unstructured_docs", [])
    total = 0
    for d in docs:
        if "content" in d:
            total += token_service.estimate_tokens(d["content"])
        elif "char_count" in d:
            total += max(1, d["char_count"] // 4)
    return total

def _get_graph_overhead_tokens(repo_id: str | None) -> int:
    # CodeGraph & Graphify require 0 LLM tokens to build (deterministic AST parsing)
    overhead = 0
    # LangGraph (PDFs) uses LLM processing to summarize communities from document text
    comm_sums = st.session_state.get("unstructured_community_summaries", {})
    if comm_sums:
        for cid, s in comm_sums.items():
            overhead += token_service.estimate_tokens(str(s))
    return overhead

st.set_page_config(
    page_title="Context Optimization Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

import textwrap

# ── Premium Dark Glassmorphism Design System ──
def _inject_design_system():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        raw_css = css_path.read_text(encoding="utf-8")
        # Strip blank lines so Markdown-it HTML block rule never triggers premature termination
        clean_css = "\n".join([l for l in raw_css.splitlines() if l.strip()])
    else:
        clean_css = ""
    font_link = '<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">'
    st.markdown(f"{font_link}\n<style>\n{clean_css}\n</style>", unsafe_allow_html=True)

_inject_design_system()

def get_secret(name: str, default: str | None = None) -> str | None:
    env_value = os.getenv(name)
    if env_value:
        return env_value
    try:
        if hasattr(st, "secrets") and st.secrets is not None:
            value = st.secrets.get(name)
            if value is not None:
                return str(value)
    except Exception:
        pass
    return default

@st.cache_resource
def services():
    data_dir_value = get_secret("CONTEXT_ENGINE_DATA_DIR") or os.getenv("CONTEXT_ENGINE_DATA_DIR")
    if data_dir_value:
        data_dir = Path(data_dir_value)
    else:
        # Fall back to /tmp/context_engine_data if running on Streamlit Cloud (read-only mount)
        if Path("/mount/src").exists() or os.getenv("STREAMLIT_SERVER_PORT"):
            data_dir = Path("/tmp/context_engine_data")
        else:
            data_dir = PROJECT_ROOT / "data"

    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    storage = LocalStorage(data_dir)
    token_service = TokenService()
    tree_sitter_service = TreeSitterService()
    codegraph_service = CodeGraphService()
    graphify_service = GraphifyService(storage=storage)
    pipeline = AnalysisPipeline(
        storage=storage,
        tree_sitter_service=tree_sitter_service,
        codegraph_service=codegraph_service,
        graphify_service=graphify_service,
        token_service=token_service,
    )
    repo_service = RepoService(storage=storage, analysis_pipeline=pipeline, max_upload_mb=200)
    llm_provider = GeminiProvider(
        api_key="",
        model="gemini-2.5-flash",
    )
    chat_service = ChatService(
        storage=storage,
        graph_retrieval_service=GraphRetrievalService(storage=storage, token_service=token_service),
        token_service=token_service,
        llm_provider=llm_provider,
        pipeline=pipeline,
    )

    return storage, pipeline, repo_service, chat_service, llm_provider, token_service

storage, pipeline, repo_service, chat_service, base_llm_provider, token_service = services()

# ── Sidebar Configurations ──

st.sidebar.markdown(textwrap.dedent("""
<div style="padding: 8px 0 4px 0; border-bottom: 1px solid rgba(99,102,241,0.15); margin-bottom: 14px;">
  <div style="font-family: 'Outfit', sans-serif; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #4b5563;">System Config</div>
  <div style="font-family: 'Outfit', sans-serif; font-size: 1.0rem; font-weight: 800; color: #c7d2fe; margin-top: 2px;">⚡ Engine Configuration</div>
</div>
"""), unsafe_allow_html=True)

provider = st.sidebar.selectbox(
    "LLM Provider",
    ["Gemini", "Groq", "OpenRouter"],
    key="llm_provider_select"
)

# API key / Credential inputs
api_key = ""
model = ""

# Handle provider default models
if provider == "Gemini":
    model = st.sidebar.selectbox("Gemini Model", ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"])
elif provider == "Groq":
    model = st.sidebar.selectbox("Groq Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])
elif provider == "OpenRouter":
    model = st.sidebar.selectbox(
        "OpenRouter Model",
        [
            "openrouter/free",
            "meta-llama/llama-3.3-70b-instruct",
            "google/gemma-4-31b-it:free",
            "openai/gpt-oss-20b:free",
            "poolside/laguna-s-2.1:free"
        ]
    )

env_secret_name = "GEMINI_API_KEY" if provider == "Gemini" else f"{provider.upper()}_API_KEY"
default_key = get_secret(env_secret_name) or get_secret("API_KEY", "") or ""
session_key_name = f"{provider.lower()}_api_key_override"

# API Key input field starts completely empty
api_key = st.sidebar.text_input(
    f"{provider} API Key",
    value=st.session_state.get(session_key_name, ""),
    placeholder="Paste your API key here...",
    type="password",
    key=f"api_key_input_{provider.lower()}"
)
st.session_state[session_key_name] = api_key
st.session_state["api_key"] = api_key
st.session_state["llm_provider"] = "Gemini" if provider == "Gemini" else f"{provider} ({model})"
st.session_state["model_name"] = model

# Instantiate dynamic LLM Provider matching sidebar configuration
def get_configured_llm_provider():
    if provider == "Gemini":
        return GeminiProvider(api_key=api_key, model=model)
    elif provider == "Groq":
        return GroqProvider(api_key=api_key, model=model)
    elif provider == "OpenRouter":
        return OpenRouterProvider(api_key=api_key, model=model)
    return base_llm_provider

# Update chat_service's llm provider dynamically
active_llm = get_configured_llm_provider()
chat_service.llm_provider = active_llm

# ── Sidebar Graph Retrieval & Search Parameters ──
st.sidebar.markdown('<hr class="glow-divider">', unsafe_allow_html=True)
st.sidebar.markdown('<div class="section-header">🗂 Graph Engine & Retrieval</div>', unsafe_allow_html=True)

# 1. Codebase Retrieval System Selectbox
current_source = st.session_state.get("harness_source_selection", "combined")
current_retrieval = st.session_state.get("harness_retrieval_method", "advanced")

if current_source == "graphify":
    default_system_idx = 1
elif current_source == "codegraph" and current_retrieval == "internal":
    default_system_idx = 2
else:
    default_system_idx = 0 # Default: Advanced Hybrid System

retrieval_system_choice = st.sidebar.selectbox(
    "Codebase Retrieval System",
    [
        "Advanced Hybrid System (Combined Dual-Graph)",
        "Graphify (Native CLI)",
        "CodeGraph (Native CLI)"
    ],
    index=default_system_idx,
    key="sidebar_retrieval_system_select",
    help="Advanced Hybrid System automatically scales budget dynamically based on repository size."
)

if "Graphify" in retrieval_system_choice:
    st.session_state["harness_source_selection"] = "graphify"
    st.session_state["harness_retrieval_method"] = "internal"
    st.session_state["harness_max_nodes"] = 8
    st.session_state["harness_graphify_mode"] = "bfs"
elif "CodeGraph" in retrieval_system_choice:
    st.session_state["harness_source_selection"] = "codegraph"
    st.session_state["harness_retrieval_method"] = "internal"
    st.session_state["harness_max_nodes"] = 8
else: # Advanced Hybrid System (Combined Dual-Graph)
    st.session_state["harness_source_selection"] = "combined"
    st.session_state["harness_retrieval_method"] = "advanced"
    st.session_state["harness_max_nodes"] = None
    st.session_state["harness_max_anchors"] = None
    st.session_state["harness_max_neighbors"] = None

st.session_state["source_preference"] = "auto"

# 2. Rectify Mode (Error Correction)
rectify_on = st.sidebar.toggle(
    "🔧 Rectify Mode (Error Correction)",
    value=st.session_state.get("harness_rectify_mode", False),
    key="sidebar_rectify_mode_ui",
    help="When enabled, the LLM produces structured <code_fix> blocks for automatic code updates."
)
st.session_state["harness_rectify_mode"] = rectify_on
if rectify_on:
    st.sidebar.info("🔧 **Rectify Mode ON** — Structured code fixes enabled.")



# Ingest ZIP / files helpers
def ingest_uploaded_zip(uploaded_file) -> RepoMetadata:
    import gc
    repo_id = uuid4().hex
    repo_name = clean_repo_name(Path(uploaded_file.name).stem)
    upload_path = storage.uploads_dir / f"{repo_id}.zip"
    source_dir = storage.repo_source_dir(repo_id)
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(uploaded_file.getbuffer())
    if source_dir.exists():
        shutil.rmtree(source_dir)
    safe_extract_zip(upload_path, source_dir)
    res = pipeline.analyze_existing(name=repo_name, source_dir=source_dir, origin="upload", repo_id=repo_id)
    gc.collect()
    return res

def ingest_uploaded_files(uploaded_files) -> RepoMetadata:
    import gc
    repo_id = uuid4().hex
    if len(uploaded_files) == 1:
        repo_name = clean_repo_name(Path(uploaded_files[0].name).stem)
    else:
        repo_name = f"uploaded_{len(uploaded_files)}_files"
    source_dir = storage.repo_source_dir(repo_id)
    source_dir.mkdir(parents=True, exist_ok=True)
    for f in uploaded_files:
        dest = source_dir / f.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.getbuffer())
    res = pipeline.analyze_existing(name=repo_name, source_dir=source_dir, origin="file_upload", repo_id=repo_id)
    gc.collect()
    return res

# ── Live Sidebar Checklist Placeholder ──
st.sidebar.markdown('<hr class="glow-divider">', unsafe_allow_html=True)
st.sidebar.markdown('<div class="section-header">📋 Agent Execution Plan</div>', unsafe_allow_html=True)
sidebar_todo_placeholder = st.sidebar.empty()

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "harness_todo" not in st.session_state:
    st.session_state["harness_todo"] = []
else:
    # Sanitize: ensure all items are dicts with 'step' and 'status' keys
    st.session_state["harness_todo"] = [
        item for item in st.session_state["harness_todo"]
        if isinstance(item, dict) and "step" in item
    ]

if "unstructured_docs" not in st.session_state:
    st.session_state["unstructured_docs"] = []
if "unstructured_graph" not in st.session_state:
    st.session_state["unstructured_graph"] = None
if "unstructured_community_summaries" not in st.session_state:
    st.session_state["unstructured_community_summaries"] = {}
if "unstructured_community_embeddings" not in st.session_state:
    st.session_state["unstructured_community_embeddings"] = {}
if "unstructured_node_embeddings" not in st.session_state:
    st.session_state["unstructured_node_embeddings"] = {}

if "fix_applied" not in st.session_state:
    st.session_state["fix_applied"] = False



# Render static or last known checklist in sidebar
def render_sidebar_todo(todo):
    with sidebar_todo_placeholder.container():
        if todo:
            for i, item in enumerate(todo):
                # Defensive: handle any malformed item in session state
                if not isinstance(item, dict):
                    continue
                status = item.get("status", "pending")
                step = item.get("step") or item.get("task") or item.get("description") or str(item)
                if status == "completed":
                    st.markdown(f"🟢 **Step {i+1}**: {step} (Complete)")
                elif status == "in_progress":
                    st.markdown(f"🟡 **Step {i+1}**: {step} (In Progress)")
                else:
                    st.markdown(f"⚪ **Step {i+1}**: {step} (Pending)")
        else:
            st.info("Harness is idle. Submit a query to trigger planning.")


render_sidebar_todo(st.session_state["harness_todo"])

# ── Main UI Layout ──

# Active context pills
_active_repo_name = st.session_state.get("repo_name", "")
_active_pdf_count = len(st.session_state.get("unstructured_docs", []))
_pills_html = ""
if _active_repo_name:
    _pills_html += f'<span class="pill">Codebase: {_active_repo_name}</span>'
if _active_pdf_count > 0:
    _pills_html += f'<span class="pill teal">{_active_pdf_count} PDF(s) Indexed</span>'

# Build model pill string
_model_label = st.session_state.get("model_name", "")
_provider_label = st.session_state.get("llm_provider", "")
_model_pill = f'<span class="pill amber">⚡ {_provider_label} · {_model_label}</span>' if _model_label else ""

st.markdown(
    textwrap.dedent(f"""
    <div class="app-header">
      <div class="app-header-logo">
        <svg width="26" height="26" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="8" cy="8" r="4" fill="#60EFFF" opacity="0.9"/>
          <circle cx="24" cy="8" r="4" fill="#818CF8" opacity="0.9"/>
          <circle cx="16" cy="24" r="4" fill="#5EEAD4" opacity="0.9"/>
          <circle cx="16" cy="14" r="2.5" fill="white" opacity="0.7"/>
          <path d="M8 8L24 8M8 8L16 24M24 8L16 24M8 8L16 14M24 8L16 14M16 24L16 14" stroke="rgba(255,255,255,0.35)" stroke-width="1.2"/>
        </svg>
      </div>
      <div class="app-header-text">
        <h1>Context Optimization Engine</h1>
        <p>▸ Agentic Graph-Augmented Retrieval &amp; Multi-Turn QA Harness</p>
      </div>
      <div class="app-status-pill">
        {_model_pill}
        {_pills_html}
      </div>
    </div>
    """),
    unsafe_allow_html=True
)

main_tabs = st.tabs(["📦  Ingest & Index", "🌐  Visualizer", "🤖  QA Harness", "📊  Token Analytics"])

# --- Tab 0: Ingest & Index Graphs ---
with main_tabs[0]:
    st.markdown(textwrap.dedent("""
    <div style="margin-bottom: 20px;">
      <div style="font-family:'Outfit',sans-serif; font-size:0.65rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#4b5563; margin-bottom:4px;">Step 1</div>
      <div style="font-family:'Outfit',sans-serif; font-size:1.25rem; font-weight:800; color:#e2e8f0; letter-spacing:-0.01em;">Load your knowledge sources</div>
      <div style="font-family:'Inter',sans-serif; font-size:0.82rem; color:#475569; margin-top:2px;">Upload a codebase or documents — the engine will build graphs, detect communities, and index everything automatically.</div>
    </div>
    """), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        SUPPORTED_EXTENSIONS = [
            "py", "pyi", "ipynb", "js", "jsx", "ts", "tsx",
            "go", "rs", "java", "c", "cpp", "cc", "cxx", "h", "hpp",
        ]
        repo_upload = st.file_uploader(
            "Upload codebase (.zip) or individual source files",
            type=SUPPORTED_EXTENSIONS + ["zip"],
            accept_multiple_files=True,
            key="codebase_uploader"
        )
        if repo_upload:
            current_files_sig = [(f.name, f.size) for f in repo_upload]
            last_sig = st.session_state.get("last_uploaded_codebase_sig")
            if current_files_sig != last_sig:
                st.session_state["last_uploaded_codebase_sig"] = current_files_sig
                with st.spinner("Extracting and building Tree-sitter CodeGraph..."):
                    try:
                        # Check if there is a zip in the uploaded files
                        zip_files = [f for f in repo_upload if f.name.lower().endswith(".zip")]
                        if zip_files:
                            repo_meta = ingest_uploaded_zip(zip_files[0])
                        else:
                            repo_meta = ingest_uploaded_files(repo_upload)
                        
                        if repo_meta.status == RepoStatus.failed or repo_meta.status.value == "failed":
                            st.error(f"❌ Ingestion failed: {repo_meta.error}")
                            st.session_state["uploaded_codebase"] = False
                            
                            # Display diagnostics logs for why it failed
                            logs = storage.load_logs(repo_meta.repo_id, limit=30)
                            if logs:
                                with st.expander("📋 Ingestion Logs & Diagnostics", expanded=True):
                                    for log in logs:
                                        level_emoji = "🔴" if log.get("level") == "error" else "🟡" if log.get("level") == "warning" else "🔵"
                                        st.write(f"{level_emoji} **[{log.get('stage', 'pipeline').upper()}]** {log.get('message')}")
                        else:
                            st.session_state.repo_id = repo_meta.repo_id
                            st.session_state["repo_name"] = repo_meta.name
                            st.session_state["uploaded_codebase"] = True
                            st.session_state["fix_applied"] = False
                            # Rerun once to make sure the metrics and other tabs update cleanly
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Ingestion failed: {e}")
        elif st.session_state.get("uploaded_codebase"):
            st.session_state["uploaded_codebase"] = False
            st.session_state["repo_id"] = None
            st.session_state["repo_name"] = None
            st.session_state["last_uploaded_codebase_sig"] = None
            st.session_state["fix_applied"] = False
            st.rerun()

        # Display metrics if already uploaded
        if st.session_state.get("uploaded_codebase") and st.session_state.get("repo_id"):
            repo_meta = storage.load_repo_metadata(st.session_state.repo_id)
            if repo_meta and repo_meta.stats:
                st.metric("Total Files", repo_meta.stats.total_files)
                st.metric("Total Lines", repo_meta.stats.total_lines)
                    
        active_id = st.session_state.get("repo_id")
        if active_id and st.session_state.get("fix_applied"):
            try:
                import io, zipfile
                def get_repo_zip(repo_id: str) -> bytes:
                    src_dir = storage.repo_source_dir(repo_id)
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                        for p in src_dir.rglob("*"):
                            if p.is_file() and not p.name.endswith(".bak"):
                                z.write(p, p.relative_to(src_dir))
                    return buf.getvalue()
                
                zip_data = get_repo_zip(active_id)
                st.download_button(
                    label="⬇️ Download Corrected Codebase (ZIP)",
                    data=zip_data,
                    file_name=f"{st.session_state.get('repo_name', 'codebase')}_corrected.zip",
                    mime="application/zip",
                    use_container_width=True,
                    help="Download the entire codebase containing applied fixes and patches."
                )
            except Exception as e:
                st.warning(f"Could not build download zip: {e}")
                
    with col2:
        pdf_uploads = st.file_uploader(
            "Upload PDF or text guidelines (.pdf, .txt, .md)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        if pdf_uploads:
            current_pdf_sig = [(f.name, f.size) for f in pdf_uploads]
            last_pdf_sig = st.session_state.get("last_uploaded_pdf_sig")
            if current_pdf_sig != last_pdf_sig:
                st.session_state["last_uploaded_pdf_sig"] = current_pdf_sig
                with st.spinner("Chunking files..."):
                    docs = process_uploaded_files(pdf_uploads)
                    current_docs = st.session_state.get("unstructured_docs", [])
                    for d in docs:
                        if not any(cd["name"] == d["name"] for cd in current_docs):
                            current_docs.append(d)
                    st.session_state["unstructured_docs"] = current_docs
                    st.session_state["uploaded_pdf"] = True
                    st.success(f"Ingested {len(docs)} document(s).")
        elif st.session_state.get("uploaded_pdf"):
            st.session_state["uploaded_pdf"] = False
            st.session_state["unstructured_docs"] = []
            st.session_state["unstructured_graph"] = None
            st.session_state["last_uploaded_pdf_sig"] = None
            st.rerun()
                
        # Action Buttons for Knowledge Graph
        all_docs = st.session_state.get("unstructured_docs", [])
        if all_docs:
            st.markdown('<div class="section-header">📄 Document Catalog</div>', unsafe_allow_html=True)
            for doc in all_docs:
                st.write(f"- **{doc['name']}** ({doc['type']}, {doc['size_kb']} KB)")
                
            louvain_res = st.slider("Louvain Resolution (community sizing)", min_value=0.5, max_value=10.0, value=1.0, step=0.1)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🕸️ Build Knowledge Graph", use_container_width=True):
                    with st.spinner("Extracting entities & relations with LLM..."):
                        try:
                            G = build_graph_from_documents(all_docs)
                            st.session_state["unstructured_graph"] = G
                            st.success("Knowledge Graph built!")
                        except Exception as e:
                            st.error(f"Failed: {e}")
            with c2:
                disabled = st.session_state.get("unstructured_graph") is None
                if st.button("🧩 Louvain Communities", use_container_width=True, disabled=disabled):
                    with st.spinner("Detecting communities..."):
                        G = st.session_state["unstructured_graph"]
                        cmap = detect_communities(G, resolution=louvain_res)
                        st.session_state["unstructured_graph"] = G
                        num_communities = len(set(cmap.values()))
                        st.success(f"Detected {num_communities} Louvain communities.")
            with c3:
                disabled = st.session_state.get("unstructured_graph") is None
                if st.button("📝 Generate Summaries", use_container_width=True, disabled=disabled):
                    with st.spinner("Summarizing communities..."):
                        try:
                            G = st.session_state["unstructured_graph"]
                            sums, embs = summarize_all_communities(G)
                            st.session_state["unstructured_community_summaries"] = sums
                            st.session_state["unstructured_community_embeddings"] = embs
                            st.success("Generated community summaries.")
                        except Exception as e:
                            st.error(f"Failed: {e}")

# --- Tab 1: Visualizer Dashboard ---
with main_tabs[1]:
    st.markdown(textwrap.dedent("""
    <div style="margin-bottom: 20px;">
      <div style="font-family:'Outfit',sans-serif; font-size:0.65rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#4b5563; margin-bottom:4px;">Interactive</div>
      <div style="font-family:'Outfit',sans-serif; font-size:1.25rem; font-weight:800; color:#e2e8f0; letter-spacing:-0.01em;">Graph Visualizer Dashboard</div>
      <div style="font-family:'Inter',sans-serif; font-size:0.82rem; color:#475569; margin-top:2px;">Explore the structure of your codebase and document knowledge graphs interactively.</div>
    </div>
    """), unsafe_allow_html=True)
    st.markdown('<div class="section-header">📂 Codebase Graph Visualizer</div>', unsafe_allow_html=True)
    repo_id = st.session_state.get("repo_id")
    if not repo_id:
        st.warning("Upload a codebase in Tab 1 to visualize the Codebase Graph.")
    else:
        view_mode = st.radio(
            "Abstraction Level:",
            ["Graphify (High-Level Architecture)", "CodeGraph (AST-Level Details)"],
            horizontal=True,
            key="codebase_visualizer_view_mode"
        )
        
        if view_mode.startswith("Graphify"):
            graph_doc = storage.load_graphify(repo_id)
            graph_title = "Graphify Network"
        else:
            graph_doc = storage.load_codegraph(repo_id)
            graph_title = "CodeGraph AST Network"
            
        if not graph_doc:
            st.error(f"Could not load {graph_title} file.")
        else:
            with st.spinner(f"Compiling {graph_title} visualization..."):
                # Map GraphDocument → NetworkX DiGraph (directed edges = call/import flow)
                G_code = nx.DiGraph()
                node_type_counts: dict[str, int] = {}

                for node in graph_doc.nodes:
                    if node.node_type == "cli_output":
                        continue
                    # Rich tooltip description: source snippet or file location
                    desc = (node.source_snippet or "")[:500]
                    if not desc:
                        if node.file_path:
                            desc = f"{node.file_path}:{node.line_start}"
                        else:
                            desc = f"{node.node_type}: {node.label}"

                    node_type_counts[node.node_type] = node_type_counts.get(node.node_type, 0) + 1
                    G_code.add_node(
                        node.node_id,
                        label=node.label,
                        type=node.node_type,
                        community_id=node.node_type,   # drives color grouping by type
                        description=desc,
                    )

                for edge in graph_doc.edges:
                    if G_code.has_node(edge.source_node) and G_code.has_node(edge.target_node):
                        G_code.add_edge(
                            edge.source_node,
                            edge.target_node,
                            relation_type=edge.edge_type,
                        )

                mc1, mc2 = st.columns(2)
                mc1.metric("Nodes", len(G_code.nodes))
                mc2.metric("Edges", len(G_code.edges))

                try:
                    code_html = graph_to_pyvis(G_code, height="650px")
                    import streamlit.components.v1 as components
                    components.html(code_html, height=670, scrolling=False)
                except Exception as e:
                    st.error(f"Could not render {graph_title}: {e}")

                # Node-type legend
                legend_items = [
                    ("module / file",      "#3B82F6",  "Blue",    ["module",  "file"]),
                    ("class / component",  "#A855F7",  "Purple",  ["class",   "component"]),
                    ("function",           "#60EFFF",  "Cyan",    ["function"]),
                    ("method",             "#34D399",  "Emerald", ["method"]),
                    ("import",             "#94A3B8",  "Slate",   ["import"]),
                    ("external_symbol",    "#64748B",  "Grey",    ["external_symbol"]),
                    ("concept",            "#F59E0B",  "Amber",   ["concept"]),
                ]
                legend_html = "<div style='display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;'>"
                for ltype, lcolor, lname, ltypes in legend_items:
                    count = sum(node_type_counts.get(t, 0) for t in ltypes)
                    if count > 0:
                        legend_html += (
                            f"<span style='display:inline-flex;align-items:center;gap:6px;"
                            f"font-size:12px;color:#e2e8f0;background:rgba(255,255,255,0.05);"
                            f"border-radius:6px;padding:3px 8px'>"
                            f"<span style='width:11px;height:11px;border-radius:50%;"
                            f"background:{lcolor};display:inline-block;flex-shrink:0'></span>"
                            f"<span style='color:{lcolor};font-weight:600'>{lname}</span>"
                            f" {ltype} <b>({count})</b></span>"
                        )
                legend_html += "</div>"
                st.markdown(legend_html, unsafe_allow_html=True)
                    
    st.markdown("---")

    # ── Section 2: Document Knowledge Graph (Louvain Communities) ──
    st.markdown("### 🕸️ Document Knowledge Graph (Louvain Communities)")
    G_pdf = st.session_state.get("unstructured_graph")
    if G_pdf is None:
        st.warning("Upload documents and build the Knowledge Graph in Tab 1 to visualize the Network.")
    else:
        with st.spinner("Compiling Knowledge Graph visualization..."):
            try:
                pdf_html = graph_to_pyvis(G_pdf, height="650px")
                import streamlit.components.v1 as components
                components.html(pdf_html, height=670, scrolling=False)
            except Exception as e:
                st.error(f"Could not render Knowledge Graph: {e}")

with main_tabs[2]:
    st.markdown(textwrap.dedent("""
    <div style="margin-bottom: 20px;">
      <div style="font-family:'Outfit',sans-serif; font-size:0.65rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#4b5563; margin-bottom:4px;">ReAct Agent Harness</div>
      <div style="font-family:'Outfit',sans-serif; font-size:1.25rem; font-weight:800; color:#e2e8f0; letter-spacing:-0.01em;">Graph-Augmented QA Loop</div>
      <div style="font-family:'Inter',sans-serif; font-size:0.82rem; color:#475569; margin-top:2px;">Ask anything about your codebase or documents. Multi-turn memory preserves context across queries.</div>
    </div>
    """), unsafe_allow_html=True)
    

    
    # Render chat history
    for msg_idx, msg in enumerate(st.session_state["chat_history"]):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(_clean_xml_from_text(msg["content"]))
            else:
                st.markdown(msg["content"])

            # ── Per-Message Retrieval System Inspector & Code Fix Panels (assistant messages only) ──
            if msg["role"] == "assistant":
                records = msg.get("retrieval_records", [])
                if records:
                    with st.expander(
                        f"🔍 Retrieval System Inspector ({len(records)} query context{'s' if len(records) > 1 else ''} sent for this chat)",
                        expanded=False
                    ):
                        for r_idx, rec in enumerate(records):
                            st.markdown(f"**Query {r_idx+1}**: `{rec.get('query', '')}` (Engine: `{rec.get('type', '')}` at {rec.get('timestamp', 'N/A')})")
                            if "LangGraph" in rec.get("type", ""):
                                st.markdown("##### 🧩 Retrieved Louvain Communities & Partial Answers")
                                per_comm = rec.get("per_comm_details", [])
                                if not per_comm:
                                    st.warning(rec.get("merged_context_prompt", "No communities retrieved."))
                                else:
                                    for item in per_comm:
                                        st.markdown(f"**Community ID: {item.get('cid', 'N/A')}** (Score: {item.get('score', 0.0):.3f})")
                                        st.write(f"**Anchors**: {item.get('anchors', [])}")
                                        st.info(item.get("summary", ""))
                                        st.success(item.get("partial_answer", ""))
                                st.markdown("##### 📝 Merged Prompt Context (Sent to LLM)")
                                st.code(rec.get("merged_context_prompt", "No context prompt available."), language="text")
                            else:
                                st.markdown("##### 🌲 Retrieved AST Codebase Subgraph & Definitions")
                                st.code(rec.get("context_retrieved", "No explicit source context retrieved."), language="python")

                fixes = _extract_code_fixes(msg["content"])
                for fix_idx, fix in enumerate(fixes):
                    fix_key = f"fix_{msg_idx}_{fix_idx}"
                    with st.expander(
                        f"🔧 Proposed Code Fix → `{fix['filepath']}`",
                        expanded=True
                    ):
                        st.markdown(f"**File:** `{fix['filepath']}`")
                        fc1, fc2 = st.columns(2)
                        with fc1:
                            st.markdown("**Original Code (to replace)**")
                            st.code(fix["original"], language="python")
                        with fc2:
                            st.markdown("**Replacement Code**")
                            st.code(fix["replacement"], language="python")

                        btn_col1, btn_col2 = st.columns([1, 1])
                        active_repo = st.session_state.get("repo_id")

                        # Apply button — patches file on disk and rebuilds CodeGraph + Graphify
                        with btn_col1:
                            apply_clicked = st.button(
                                "✅ Apply Fix & Rebuild Graphs",
                                key=f"apply_{fix_key}",
                                use_container_width=True,
                                disabled=not bool(active_repo),
                                help="Applies the fix to disk, then automatically rebuilds CodeGraph and Graphify."
                            )
                        if apply_clicked and active_repo:
                            with st.spinner(f"Applying fix to `{fix['filepath']}` and rebuilding graphs..."):
                                result = chat_service.apply_rectification(
                                    repo_id=active_repo,
                                    file_path=fix["filepath"],
                                    original_code=fix["original"],
                                    replacement_code=fix["replacement"],
                                )
                            if result.get("status") == "success":
                                st.success(
                                    f"✅ Fix applied to `{fix['filepath']}`. "
                                    f"Backup saved as `{result.get('backup_path')}`. "
                                    "CodeGraph & Graphify rebuilt."
                                )
                                st.session_state["fix_applied"] = True
                                # Store corrected content for download
                                st.session_state[f"corrected_{fix_key}"] = (
                                    fix["filepath"],
                                    result.get("new_content", ""),
                                )
                            else:
                                st.error(f"❌ Fix failed: {result.get('error', 'Unknown error')}")
                                st.info(
                                    "**Reason:** " + result.get("error", "") +
                                    "\n\nThis usually means the original code block was not found verbatim "
                                    "in the file. The LLM may have added/removed whitespace. "
                                    "Try enabling **Rectify Mode** in the retrieval settings so the LLM "
                                    "generates more precise `<original_code>` blocks."
                                )

                        # Download button — available after successful apply
                        with btn_col2:
                            corrected = st.session_state.get(f"corrected_{fix_key}")
                            if corrected:
                                dl_path, dl_content = corrected
                                st.download_button(
                                    label="⬇️ Download Corrected File",
                                    data=dl_content,
                                    file_name=Path(dl_path).name,
                                    mime="text/plain",
                                    key=f"dl_{fix_key}",
                                    use_container_width=True,
                                )
                            else:
                                st.button(
                                    "⬇️ Download Corrected File",
                                    key=f"dl_disabled_{fix_key}",
                                    disabled=True,
                                    use_container_width=True,
                                    help="Apply the fix first to enable download."
                                )



    # Chat Input
    if user_query := st.chat_input("Ask a question mapping codebase dependencies and unstructured rules..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Placeholders for live output
        live_thought = st.empty()
        chat_log_placeholder = st.empty()
        
        # Callback for live updating during Harness run
        def update_ui(history, todo):
            render_sidebar_todo(todo)
            with chat_log_placeholder.container():
                st.markdown("### ⚙️ Live Perception-Action-Observation Log")
                for idx, step in enumerate(history):
                    thought_snippet = step.get('thought', '')[:60] + "..." if len(step.get('thought', '')) > 60 else step.get('thought', '')
                    with st.expander(f"Iteration {idx+1}: {thought_snippet or 'Running...'} (Tool: {step.get('tool')})", expanded=(idx == len(history)-1)):
                        st.write(f"**Thought:** {step.get('thought')}")
                        st.write(f"**Action:** Call `{step.get('tool')}` with `{step.get('tool_input')}`")
                        st.code(step.get('observation'), language="text")

        # Capture retrieval history count before execution
        hist_before_len = len(st.session_state.get("retrieval_history", []))

        with st.spinner("Initializing central agentic harness loop..."):
            harness = AgentHarness(chat_service=chat_service, llm_provider=active_llm)
            try:
                # Determine available assets and max iterations:
                #   Both assets → 6 iters (query both graphs, merge)
                #   One asset   → 4 iters (query one graph)
                #   No assets   → 1 iter  (harness fast-paths to direct LLM answer)
                has_pdf  = bool(st.session_state.get("uploaded_pdf"))
                has_code = bool(st.session_state.get("uploaded_codebase"))
                if has_pdf and has_code:
                    max_iters = 6
                elif has_pdf or has_code:
                    max_iters = 4
                else:
                    max_iters = 1

                pref = st.session_state.get("source_preference", "auto").replace(" ", "_").lower()
                result = harness.execute(
                    user_query, 
                    max_iterations=max_iters, 
                    source_preference=pref, 
                    callback=update_ui,
                    chat_history=st.session_state.get("chat_history", [])
                )
                final_answer = result["final_answer"]
                history_log = result["history"]

            except Exception as e:
                final_answer = f"Harness loop crashed with error: {str(e)}"
                history_log = [{"thought": "Harness execution failed", "tool": "none", "tool_input": "{}", "observation": str(e)}]

        hist_after = st.session_state.get("retrieval_history", [])
        new_retrievals = hist_after[:len(hist_after) - hist_before_len] if len(hist_after) > hist_before_len else []

        # Save to chat history
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": final_answer,
            "harness_history": history_log,
            "retrieval_records": new_retrievals,
            "timestamp": time.strftime("%H:%M:%S")
        })
        st.session_state["harness_todo"] = st.session_state.get("harness_todo", [])
        st.rerun()

# --- Tab 3: Token Analytics ---
with main_tabs[3]:
    st.markdown(textwrap.dedent("""
    <div style="margin-bottom: 20px;">
      <div style="font-family:'Outfit',sans-serif; font-size:0.65rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#4b5563; margin-bottom:4px;">Diagnostics</div>
      <div style="font-family:'Outfit',sans-serif; font-size:1.25rem; font-weight:800; color:#e2e8f0; letter-spacing:-0.01em;">Token Analytics & Context Savings</div>
      <div style="font-family:'Inter',sans-serif; font-size:0.82rem; color:#475569; margin-top:2px;">Understand how much context each retrieval method uses and how efficiently the engine compresses your codebase.</div>
    </div>
    """), unsafe_allow_html=True)

    active_repo = st.session_state.get("repo_id")
    codebase_tokens = _get_active_codebase_tokens(active_repo)
    pdf_tokens = _get_active_pdf_tokens()
    asset_baseline_single = codebase_tokens + pdf_tokens
    overhead_tokens = _get_graph_overhead_tokens(active_repo)

    # ── Top Inventory Section ──
    st.markdown("### 📦 Asset Inventory & Graph Construction Overhead")
    ac1, ac2, ac3, ac4 = st.columns(4)
    ac1.metric("Active Codebase Tokens", f"{codebase_tokens:,}")
    ac2.metric("Active PDF/Guideline Tokens", f"{pdf_tokens:,}")
    ac3.metric("Single-Query Raw Baseline", f"{asset_baseline_single:,}")
    ac4.metric("Graph Overhead Tokens", f"{overhead_tokens:,}", help="Tokens consumed generating community summaries & AST schemas.")

    st.markdown("---")

    # ── Per-Chat Conversation Breakdown ──
    st.markdown("### 💬 Per-Chat Conversation Token Breakdown")
    chat_history = st.session_state.get("chat_history", [])
    
    # Extract paired chat queries & responses
    chat_details = []
    pair_idx = 1
    total_chat_opt_tokens = 0
    total_prompt_tokens = 0

    for idx, m in enumerate(chat_history):
        if m.get("role") == "user":
            u_prompt = m.get("content", "")
            p_tokens = token_service.estimate_tokens(u_prompt)

            a_msg = chat_history[idx + 1] if (idx + 1 < len(chat_history) and chat_history[idx + 1].get("role") == "assistant") else {}
            a_answer = a_msg.get("content", "")
            r_tokens = token_service.estimate_tokens(a_answer)

            recs = a_msg.get("retrieval_records", [])
            c_tokens = 0
            for r in recs:
                t_cnt = r.get("context_tokens", 0)
                if t_cnt == 0:
                    ctx_str = r.get("context_retrieved") or r.get("merged_context_prompt") or r.get("context") or ""
                    if ctx_str:
                        t_cnt = token_service.estimate_tokens(ctx_str)
                c_tokens += t_cnt
                
            # Extract unique systems used
            sys_set = set()
            for r in recs:
                stype = r.get("type", "Graph")
                smethod = r.get("retrieval_method", "")
                if "Advanced" in smethod or smethod == "advanced":
                    sys_set.add(f"{stype} Advanced")
                elif "Internal" in smethod or smethod == "internal":
                    sys_set.add(f"{stype} CLI")
                else:
                    sys_set.add(stype)
            sys_used = ", ".join(sys_set) if sys_set else "Direct Harness"

            single_opt = c_tokens
            single_base = asset_baseline_single if asset_baseline_single > 0 else max(100, c_tokens * 10)
            savings = (1.0 - (single_opt / single_base)) * 100.0 if single_base > 0 else 0.0

            total_chat_opt_tokens += c_tokens
            total_prompt_tokens += p_tokens

            chat_details.append({
                "turn": pair_idx,
                "query": u_prompt,
                "sys_used": sys_used,
                "p_tokens": p_tokens,
                "c_tokens": c_tokens,
                "r_tokens": r_tokens,
                "single_opt": single_opt,
                "single_base": single_base,
                "savings": savings,
                "recs": recs
            })
            pair_idx += 1

    if not chat_details:
        st.info("No chat queries executed yet. Submit a query in Tab 2 to track per-chat token consumption!")
    else:
        # Render Table Summary
        table_rows = []
        for cd in chat_details:
            table_rows.append({
                "Chat Turn": f"Chat #{cd['turn']}",
                "Query": cd["query"][:45] + "..." if len(cd["query"]) > 45 else cd["query"],
                "Retrieval System": cd["sys_used"],
                "Derived Context Tokens": cd["c_tokens"],
                "Raw Asset Baseline": cd["single_base"],
                "Context Savings %": f"{cd['savings']:.1f}%"
            })

        import pandas as pd
        df_chats = pd.DataFrame(table_rows)
        st.dataframe(df_chats, use_container_width=True)

        # Expandable Detail Cards
        for cd in chat_details:
            with st.expander(f"💬 Chat #{cd['turn']}: `{cd['query'][:60]}` ({cd['sys_used']})", expanded=False):
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("Prompt Tokens", f"{cd['p_tokens']:,}")
                cc2.metric("Derived Context Tokens", f"{cd['c_tokens']:,}")
                cc3.metric("Response Tokens", f"{cd['r_tokens']:,}")
                cc4.metric("Context Reduction", f"{cd['savings']:.1f}%")
                st.caption(f"**Raw Asset Baseline Context**: {cd['single_base']:,} tokens  |  **Derived Graph Context**: {cd['c_tokens']:,} tokens")

    st.markdown("---")

    # ── Cumulative Total Comparison ──
    st.markdown("### 📈 Cumulative Multi-Turn Comparison & Savings")
    num_chats = len(chat_details)
    
    if num_chats == 0:
        cum_baseline = asset_baseline_single  # 0 if nothing uploaded yet
        cum_optimized = overhead_tokens
        cum_savings_pct = 0.0
    else:
        # Baseline doubles/multiplies across N chat turns
        cum_baseline = num_chats * asset_baseline_single
        cum_optimized = overhead_tokens + total_chat_opt_tokens
        cum_savings_pct = (1.0 - (cum_optimized / cum_baseline)) * 100.0 if cum_baseline > 0 else 0.0

    col_base, col_opt, col_pct = st.columns(3)
    with col_base:
        st.metric("Total Raw Asset Baseline", f"{cum_baseline:,} tokens")
        st.caption(f"Raw Codebase ({codebase_tokens:,}) + PDF ({pdf_tokens:,}) across {max(1, num_chats)} chat turn(s).")
    with col_opt:
        st.metric("Total Graph-Derived Context", f"{cum_optimized:,} tokens")
        st.caption("Graph Construction Overhead + Derived Subgraph Contexts across all chats.")
    with col_pct:
        st.metric("Net Context Reduction Efficiency", f"{cum_savings_pct:.1f}%")
        st.caption(f"Net optimization efficiency across {num_chats} chat turn(s).")

    # Draw Cumulative Bar Chart
    import pandas as pd
    chart_df = pd.DataFrame({
        "Context Approach": [
            f"Raw Baseline ({max(1, num_chats)} Chat Turns)",
            "Graph-Optimized (Overhead + All Chats)"
        ],
        "Total Tokens": [cum_baseline, cum_optimized]
    })
    st.bar_chart(chart_df, x="Context Approach", y="Total Tokens", color="#6366f1")
