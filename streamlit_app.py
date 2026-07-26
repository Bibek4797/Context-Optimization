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
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Streamlit Cloud naming conflict cleanup for 'app' namespace
if "app" in sys.modules:
    app_module = sys.modules["app"]
    app_file = getattr(app_module, "__file__", "") or ""
    if not app_file or "backend" not in str(app_file):
        del sys.modules["app"]

from app.models.schemas import GraphDocument, QueryRecord, RepoMetadata, TreeNode
from app.services.analysis_pipeline import AnalysisPipeline
from app.services.chat_service import ChatService
from app.services.codegraph_service import CodeGraphService
from app.services.file_utils import clean_repo_name, safe_extract_zip
from app.services.graph_retrieval_service import GraphRetrievalService
from app.services.graphify_service import GraphifyService
from app.services.llm.gemini import GeminiProvider
from app.services.llm.openai_compatible import GroqProvider, OpenRouterProvider
from app.services.llm.bedrock import BedrockProvider
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
    fixes = []
    pattern = re.compile(
        r"<code_fix>\s*"
        r"<filepath>(?P<filepath>[^<]+)</filepath>\s*"
        r"<original_code>(?P<original>[\s\S]*?)</original_code>\s*"
        r"<replacement_code>(?P<replacement>[\s\S]*?)</replacement_code>\s*"
        r"</code_fix>",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        fixes.append({
            "filepath": m.group("filepath").strip(),
            "original": m.group("original").strip(),
            "replacement": m.group("replacement").strip(),
        })
    return fixes


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

# Custom styles injection for high-end dark radial gradient
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 96% !important;
    }
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0d0e15 0%, #050608 100%) !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
    }
    /* ── App Header ── */
    .app-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 18px 0 14px 0;
        border-bottom: 1px solid rgba(99, 102, 241, 0.18);
        margin-bottom: 22px;
    }
    .app-header-logo {
        width: 42px;
        height: 42px;
        background: linear-gradient(135deg, #6366f1, #0d9488);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        flex-shrink: 0;
    }
    .app-header-text h1 {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #6366F1 0%, #60EFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2 !important;
    }
    .app-header-text p {
        margin: 2px 0 0 0 !important;
        font-size: 0.78rem !important;
        color: #718096 !important;
        letter-spacing: 0.04em;
    }
    .app-status-pill {
        margin-left: auto;
        display: flex;
        gap: 8px;
        align-items: center;
    }
    .pill {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #818cf8;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        white-space: nowrap;
    }
    .pill.teal {
        background: rgba(13, 148, 136, 0.1);
        border-color: rgba(13, 148, 136, 0.3);
        color: #5eead4;
    }
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: rgba(13, 14, 21, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    /* ── Tabs ── */
    div[data-baseweb="tab-list"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-radius: 12px !important;
        padding: 6px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        gap: 8px !important;
        margin-bottom: 20px !important;
    }
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #a0aec0 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 22px !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    button[data-baseweb="tab"]:hover {
        background: rgba(99, 102, 241, 0.07) !important;
        color: #c7d2fe !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(99, 102, 241, 0.14) !important;
        color: #818cf8 !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
    }
    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 14px 16px !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        color: #e2e8f0 !important;
    }
    /* ── Misc helpers ── */
    .todo-completed { color: #10B981 !important; font-weight: 600; }
    .todo-inprogress { color: #60EFFF !important; font-style: italic; }
    .todo-pending { color: #a0aec0 !important; }
    .dark-teal-highlight {
        background: rgba(13, 148, 136, 0.1);
        border-left: 4px solid #0d9488;
        padding: 10px;
        color: #5eead4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def get_secret(name: str, default: str | None = None) -> str | None:
    env_value = os.getenv(name)
    if env_value:
        return env_value
    local_secrets = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    home_secrets = Path.home() / ".streamlit" / "secrets.toml"
    if not local_secrets.exists() and not home_secrets.exists():
        return default
    try:
        value = st.secrets.get(name)
        return str(value) if value else default
    except Exception:
        return default

@st.cache_resource
def services():
    data_dir_value = get_secret("CONTEXT_ENGINE_DATA_DIR") or os.getenv("CONTEXT_ENGINE_DATA_DIR")
    data_dir = Path(data_dir_value) if data_dir_value else PROJECT_ROOT / "data"
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

st.sidebar.title("⚙️ Engine Configuration")

provider = st.sidebar.selectbox(
    "LLM Provider",
    ["Gemini", "Groq", "OpenRouter", "Bedrock"],
    key="llm_provider_select"
)

# API key / Credential inputs
api_key = ""
aws_access_key = ""
aws_secret_key = ""
aws_region = "us-east-1"
model = ""

# Handle provider default models
if provider == "Gemini":
    model = st.sidebar.selectbox("Gemini Model", ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"])
elif provider == "Groq":
    model = st.sidebar.selectbox("Groq Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])
elif provider == "OpenRouter":
    model = st.sidebar.selectbox("OpenRouter Model", ["meta-llama/llama-3.3-70b-instruct:free"])
elif provider == "Bedrock":
    model = st.sidebar.selectbox("Bedrock Model", ["anthropic.claude-3-5-sonnet-20241022-v2:0", "amazon.nova-lite-v1:0"])

if provider != "Bedrock":
    default_key = ""
    session_key_name = f"{provider.lower()}_api_key_override"
    if session_key_name not in st.session_state:
        st.session_state[session_key_name] = default_key
        
    api_key = st.sidebar.text_input(
        f"{provider} API Key",
        value=st.session_state[session_key_name],
        type="password",
        key=f"api_key_input_{provider.lower()}"
    )
    st.session_state[session_key_name] = api_key
    st.session_state["api_key"] = api_key
    st.session_state["llm_provider"] = "Gemini" if provider == "Gemini" else f"{provider} ({model})"
    st.session_state["model_name"] = model
else:
    default_access = ""
    default_secret = ""
    default_region = "us-east-1"
    
    if "bedrock_access_key_override" not in st.session_state:
        st.session_state["bedrock_access_key_override"] = default_access
    if "bedrock_secret_key_override" not in st.session_state:
        st.session_state["bedrock_secret_key_override"] = default_secret
    if "bedrock_region_override" not in st.session_state:
        st.session_state["bedrock_region_override"] = default_region
        
    aws_access_key = st.sidebar.text_input("AWS Access Key ID", value=st.session_state["bedrock_access_key_override"], type="password")
    aws_secret_key = st.sidebar.text_input("AWS Secret Access Key", value=st.session_state["bedrock_secret_key_override"], type="password")
    aws_region = st.sidebar.text_input("AWS Region", value=st.session_state["bedrock_region_override"])
    
    st.session_state["bedrock_access_key_override"] = aws_access_key
    st.session_state["bedrock_secret_key_override"] = aws_secret_key
    st.session_state["bedrock_region_override"] = aws_region
    
    st.session_state["aws_access_key"] = aws_access_key
    st.session_state["aws_secret_key"] = aws_secret_key
    st.session_state["aws_region"] = aws_region
    st.session_state["llm_provider"] = "Amazon Bedrock"
    st.session_state["model_name"] = model

# Instantiate dynamic LLM Provider matching sidebar configuration
def get_configured_llm_provider():
    if provider == "Gemini":
        return GeminiProvider(api_key=api_key, model=model)
    elif provider == "Groq":
        return GroqProvider(api_key=api_key, model=model)
    elif provider == "OpenRouter":
        return OpenRouterProvider(api_key=api_key, model=model)
    elif provider == "Bedrock":
        return BedrockProvider(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_region=aws_region,
            model=model
        )
    return base_llm_provider

# Update chat_service's llm provider dynamically
active_llm = get_configured_llm_provider()
chat_service.llm_provider = active_llm

# ── Sidebar Graph Retrieval & Search Parameters ──
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Graph Engine & Retrieval Parameters")

# 1. Codebase Graph System
graph_system = st.sidebar.selectbox(
    "Codebase Graph System",
    ["CodeGraph (AST-Level Details)", "Graphify (High-Level Architecture)"],
    index=0 if st.session_state.get("harness_source_selection") != "graphify" else 1,
    key="sidebar_graph_system_select",
    help="Select whether query context is extracted from CodeGraph (AST nodes) or Graphify (architecture graph)."
)
st.session_state["harness_source_selection"] = "graphify" if "Graphify" in graph_system else "codegraph"

# 2. Retrieval Engine
retrieval_engine = st.sidebar.selectbox(
    "Retrieval Engine",
    ["Native/Internal CLI Engine", "Advanced Hybrid Scoring (BM25 + PageRank)"],
    index=0 if st.session_state.get("harness_retrieval_method") != "advanced" else 1,
    key="sidebar_retrieval_engine_select",
    help="Native CLI uses command-line query tools. Advanced Hybrid uses PageRank + BM25 search."
)
st.session_state["harness_retrieval_method"] = "advanced" if "Advanced" in retrieval_engine else "internal"

# 3. Source Priority Strategy
source_pref = st.sidebar.selectbox(
    "Source Priority Strategy",
    ["Auto (Smart Selection)", "Code-First (CodeGraph / Graphify)", "PDF-First (LangGraph Document RAG)"],
    index=0,
    key="sidebar_source_pref_select",
    help="Sets routing priority when both codebase and PDF documents are indexed."
)
if "Code-First" in source_pref:
    st.session_state["source_preference"] = "code_first"
elif "PDF-First" in source_pref:
    st.session_state["source_preference"] = "pdf_first"
else:
    st.session_state["source_preference"] = "auto"

# 4. Rectify Mode (Error Correction)
rectify_on = st.sidebar.toggle(
    "🔧 Rectify Mode (Error Correction)",
    value=st.session_state.get("harness_rectify_mode", False),
    key="sidebar_rectify_mode_ui",
    help="When enabled, the LLM produces structured <code_fix> blocks for automatic code updates."
)
st.session_state["harness_rectify_mode"] = rectify_on
if rectify_on:
    st.sidebar.info("🔧 **Rectify Mode ON** — Structured code fixes enabled.")

# 5. Sliders: Max Nodes & Max Neighbors / Traversal Strategy
st.session_state["harness_max_nodes"] = st.sidebar.slider(
    "Max Nodes (Context Budget)",
    min_value=2,
    max_value=30,
    value=st.session_state.get("harness_max_nodes", 8),
    key="sidebar_max_nodes_ui"
)

if st.session_state["harness_source_selection"] == "graphify":
    graphify_mode_label = st.sidebar.radio(
        "Graphify Traversal Strategy",
        ["Broad Architecture (BFS)", "Deep Execution Path (DFS)"],
        index=0 if st.session_state.get("harness_graphify_mode") != "dfs" else 1,
        key="sidebar_graphify_mode_ui"
    )
    st.session_state["harness_graphify_mode"] = "dfs" if "DFS" in graphify_mode_label else "bfs"
    st.session_state["harness_max_neighbors"] = 4
else:
    st.session_state["harness_max_neighbors"] = st.sidebar.slider(
        "Max Neighbors (AST Hops)",
        min_value=1,
        max_value=15,
        value=st.session_state.get("harness_max_neighbors", 4),
        key="sidebar_max_neighbors_ui"
    )



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
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Agent Execution Plan")
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

# Auto-restore active repository if stored on disk and not yet in session state
all_saved_repos = storage.list_repos() if hasattr(storage, "list_repos") else []
if not st.session_state.get("repo_id") and all_saved_repos:
    latest_repo = all_saved_repos[0]
    st.session_state["repo_id"] = latest_repo.repo_id
    st.session_state["repo_name"] = latest_repo.name
    st.session_state["uploaded_codebase"] = True

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

st.markdown(
    f"""
    <div class="app-header">
      <div class="app-header-logo">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="6" cy="6" r="3" fill="#60EFFF"/>
          <circle cx="18" cy="6" r="3" fill="#818CF8"/>
          <circle cx="12" cy="18" r="3" fill="#5EEAD4"/>
          <path d="M6 6L18 6M6 6L12 18M18 6L12 18" stroke="rgba(255,255,255,0.6)" stroke-width="1.5"/>
        </svg>
      </div>
      <div class="app-header-text">
        <h1>Context Optimization Engine</h1>
        <p>Agentic Graph-Augmented Retrieval &amp; Multi-Turn QA Harness</p>
      </div>
      <div class="app-status-pill">{_pills_html}</div>
    </div>
    """,
    unsafe_allow_html=True
)

main_tabs = st.tabs(["Ingest & Index Graphs", "Visualizer Dashboard", "Master Loop QA", "Token Analytics"])

# --- Tab 0: Ingest & Index Graphs ---
with main_tabs[0]:
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
                        st.success(f"✅ Successfully built CodeGraph for repository: {repo_meta.name}")
                        st.metric("Total Files", repo_meta.stats.total_files)
                        st.metric("Total Lines", repo_meta.stats.total_lines)
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")
                    
        # Check active & stored repositories
        all_repos = storage.list_repos() if hasattr(storage, "list_repos") else []
        if all_repos:
            repo_options = {f"{r.name} ({r.stats.total_files} files)": r.repo_id for r in all_repos}
            current_id = st.session_state.get("repo_id", all_repos[0].repo_id)
            current_idx = list(repo_options.values()).index(current_id) if current_id in repo_options.values() else 0
            
            selected_label = st.selectbox(
                "📂 Active Indexed Codebase",
                options=list(repo_options.keys()),
                index=current_idx,
                key="active_repo_selector_ui",
                help="Select an existing indexed codebase to visualize or query."
            )
            selected_id = repo_options[selected_label]
            if selected_id != st.session_state.get("repo_id"):
                st.session_state["repo_id"] = selected_id
                meta = storage.load_repo_metadata(selected_id)
                if meta:
                    st.session_state["repo_name"] = meta.name
                    st.session_state["uploaded_codebase"] = True
                st.rerun()
                
            # Display active repository status & log tail for diagnostic clarity
            meta = storage.load_repo_metadata(selected_id)
            if meta:
                st.write(f"**Ingestion Status**: `{meta.status}`")
                if meta.error:
                    st.error(f"**Ingestion Error**: {meta.error}")
                
                logs = storage.load_logs(selected_id, limit=30)
                if logs:
                    with st.expander("📋 Pipeline Ingestion Logs / Diagnostics", expanded=meta.status == "failed"):
                        for log in logs:
                            level_emoji = "🔴" if log.get("level") == "error" else "🟡" if log.get("level") == "warning" else "🔵"
                            st.write(f"{level_emoji} **[{log.get('stage', 'pipeline').upper()}]** {log.get('message')}")
                
    with col2:
        pdf_uploads = st.file_uploader(
            "Upload PDF or text guidelines (.pdf, .txt, .md)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        if pdf_uploads:
            with st.spinner("Chunking files..."):
                docs = process_uploaded_files(pdf_uploads)
                current_docs = st.session_state.get("unstructured_docs", [])
                for d in docs:
                    if not any(cd["name"] == d["name"] for cd in current_docs):
                        current_docs.append(d)
                st.session_state["unstructured_docs"] = current_docs
                st.session_state["uploaded_pdf"] = True
                st.success(f"Ingested {len(docs)} document(s).")
                
        # Action Buttons for LangGraph
        all_docs = st.session_state.get("unstructured_docs", [])
        if all_docs:
            st.markdown("### Document Catalog")
            for doc in all_docs:
                st.write(f"- **{doc['name']}** ({doc['type']}, {doc['size_kb']} KB)")
                
            louvain_res = st.slider("Louvain Resolution (community sizing)", min_value=0.5, max_value=10.0, value=1.0, step=0.1)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🕸️ Build LangGraph", use_container_width=True):
                    with st.spinner("Extracting entities & relations with LLM..."):
                        try:
                            G = build_graph_from_documents(all_docs)
                            st.session_state["unstructured_graph"] = G
                            st.success("LangGraph built!")
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
    st.subheader("🌐 Visualizer Dashboard")
    
    # ── Section 1: Codebase Graph Visualizer (Full Width) ──
    st.markdown("### 📂 Codebase Graph Visualizer")
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

    # ── Section 2: LangGraph Unstructured Community Network (Full Width) ──
    st.markdown("### 🕸️ LangGraph Unstructured Community Network")
    G_pdf = st.session_state.get("unstructured_graph")
    if G_pdf is None:
        st.warning("Upload documents and build the LangGraph in Tab 1 to visualize the Network.")
    else:
        with st.spinner("Compiling LangGraph visualization..."):
            try:
                pdf_html = graph_to_pyvis(G_pdf, height="650px")
                import streamlit.components.v1 as components
                components.html(pdf_html, height=670, scrolling=False)
            except Exception as e:
                st.error(f"Could not render LangGraph: {e}")

with main_tabs[2]:
    st.subheader("💬 stateless Master Loop QA")
    
    # 🔍 Retrieval Inspector (Graph Context & Prompts)
    retrieval_hist = st.session_state.get("retrieval_history", [])
    with st.expander("🔍 Retrieval Inspector (Graph Context & Prompts)", expanded=True if retrieval_hist else False):
        if not retrieval_hist:
            st.info("No queries executed yet. Submit a query in chat below to inspect the exact retrieved graph subgraphs, community summaries, and prompt context.")
        else:
            if len(retrieval_hist) == 1:
                selected_idx = 0
            else:
                options = [f"[{h['timestamp']}] {h['type']}: {h['query'][:50]}..." for h in retrieval_hist]
                selected_opt = st.selectbox("Select query record to inspect", options, index=0)
                selected_idx = options.index(selected_opt)

            latest = retrieval_hist[selected_idx]
            st.markdown(f"**Inspecting Query**: `{latest['query']}` (Type: `{latest['type']}` at {latest.get('timestamp', 'N/A')})")
            
            if "LangGraph" in latest.get("type", ""):
                st.markdown("### 🧩 Retrieved Louvain Communities & Partial Answers")
                per_comm = latest.get("per_comm_details", [])
                if not per_comm:
                    st.warning(latest.get("merged_context_prompt", "No communities retrieved."))
                else:
                    for item in per_comm:
                        st.markdown(f"#### 🌐 Community ID: {item.get('cid', 'N/A')} (Score: {item.get('score', 0.0):.3f})")
                        st.write(f"**Anchors**: {item.get('anchors', [])}")
                        st.write(f"**Community Summary**:")
                        st.info(item.get("summary", ""))
                        st.write(f"**Intermediate Partial Answer**:")
                        st.success(item.get("partial_answer", ""))
                
                st.markdown("### 📝 Merged Prompt Context (Sent to LLM)")
                st.code(latest.get("merged_context_prompt", "No context prompt available."), language="text")
                
            else:
                st.markdown("### 🌲 Retrieved AST Codebase Subgraph & Definitions")
                st.code(latest.get("context_retrieved", "No explicit source context retrieved."), language="python")
                st.write("**Synthesized Answer / Status**:")
                st.success(latest.get("answer", "No answer recorded."))
    
    # Render chat history
    for msg_idx, msg in enumerate(st.session_state["chat_history"]):
        with st.chat_message(msg["role"]):
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
                result = harness.execute(user_query, max_iterations=max_iters, source_preference=pref, callback=update_ui)
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
    st.subheader("📊 Multi-Turn Token Analytics & Context Savings")

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
