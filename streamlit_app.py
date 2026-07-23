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
        if "char_count" in d:
            total += d["char_count"] // 4
        elif "content" in d:
            total += token_service.estimate_tokens(d["content"])
    return total

def _get_graph_overhead_tokens(repo_id: str | None) -> int:
    overhead = 0
    comm_sums = st.session_state.get("unstructured_community_summaries", {})
    if comm_sums:
        for cid, s in comm_sums.items():
            overhead += token_service.estimate_tokens(str(s))
    if repo_id:
        try:
            summary = storage.load_token_summary(repo_id)
            if summary and summary.stages:
                for stage_name, tm in summary.stages.items():
                    overhead += tm.tokens
        except Exception:
            pass
    return overhead

st.set_page_config(
    page_title="Harness Execution Engine",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles injection for high-end dark radial gradient
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
    .block-container {
        padding-top: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
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
    .title-gradient {
        background: linear-gradient(135deg, #6366F1 0%, #60EFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 25px !important;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(13, 14, 21, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    div[data-baseweb="tab-list"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-radius: 12px !important;
        padding: 6px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        gap: 8px !important;
    }
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #a0aec0 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.2s ease-in-out !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(99, 102, 241, 0.12) !important;
        color: #6366f1 !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
    }
    .todo-completed {
        color: #10B981 !important;
        font-weight: 600;
    }
    .todo-inprogress {
        color: #60EFFF !important;
        font-style: italic;
    }
    .todo-pending {
        color: #a0aec0 !important;
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

    return storage, pipeline, repo_service, chat_service, base_llm_provider, token_service

storage, pipeline, repo_service, chat_service, base_llm_provider, token_service = services()

# ── Sidebar Configurations (The Brain & Keys) ──

st.sidebar.title("🧠 Configuration & Brain")

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

# Ingest ZIP / files helpers
def ingest_uploaded_zip(uploaded_file) -> RepoMetadata:
    repo_id = uuid4().hex
    repo_name = clean_repo_name(Path(uploaded_file.name).stem)
    upload_path = storage.uploads_dir / f"{repo_id}.zip"
    source_dir = storage.repo_source_dir(repo_id)
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(uploaded_file.getbuffer())
    if source_dir.exists():
        shutil.rmtree(source_dir)
    safe_extract_zip(upload_path, source_dir)
    return pipeline.analyze_existing(name=repo_name, source_dir=source_dir, origin="upload", repo_id=repo_id)

def ingest_uploaded_files(uploaded_files) -> RepoMetadata:
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
    return pipeline.analyze_existing(name=repo_name, source_dir=source_dir, origin="file_upload", repo_id=repo_id)

# ── Live Sidebar Brain Checklist Placeholder ──
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Execution Plan (Brain)")
sidebar_todo_placeholder = st.sidebar.empty()

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "harness_todo" not in st.session_state:
    st.session_state["harness_todo"] = []
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

# Render static or last known checklist in sidebar
def render_sidebar_todo(todo):
    with sidebar_todo_placeholder.container():
        if todo:
            for i, item in enumerate(todo):
                status = item["status"]
                step = item["step"]
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

st.markdown('<h1 class="title-gradient">🕸️ Graph-Based Context Optimization for LLMs</h1>', unsafe_allow_html=True)

main_tabs = st.tabs(["📤 Ingest & Index Graphs", "🌐 Visualizer Dashboard", "💬 Master Loop QA", "📊 Token Analytics"])

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
                        st.session_state.repo_id = repo_meta.repo_id
                        st.success(f"Successfully built CodeGraph for repository: {repo_meta.name}")
                    else:
                        repo_meta = ingest_uploaded_files(repo_upload)
                        st.session_state.repo_id = repo_meta.repo_id
                        st.success(f"Successfully built CodeGraph for {len(repo_upload)} uploaded files.")
                    
                    st.metric("Total Python Files", repo_meta.stats.python_files)
                    st.metric("Total Python Lines", repo_meta.stats.python_lines)
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                    
        # Check active repository
        active_repo_id = st.session_state.get("repo_id")
        if active_repo_id:
            meta = storage.load_repo_metadata(active_repo_id)
            if meta:
                st.info(f"Active repository: **{meta.name}** (ID: {active_repo_id})")
                
    with col2:
        st.subheader("📄 Ingest Unstructured PDFs (Phase 2)")
        pdf_uploads = st.file_uploader("Upload PDF or text guidelines (.pdf, .txt, .md)", type=["pdf", "txt", "md"], accept_multiple_files=True)
        if pdf_uploads:
            with st.spinner("Chunking files..."):
                docs = process_uploaded_files(pdf_uploads)
                current_docs = st.session_state.get("unstructured_docs", [])
                for d in docs:
                    if not any(cd["name"] == d["name"] for cd in current_docs):
                        current_docs.append(d)
                st.session_state["unstructured_docs"] = current_docs
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
    
    # ⚙️ Retrieval & Search Parameters Configurator
    with st.expander("⚙️ Codebase Retrieval & Search Parameters", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            source_selection_label = st.selectbox(
                "Default Retrieval System",
                ["CodeGraph (AST-Level Details)", "Graphify (High-Level Architecture)"],
                index=0,
                key="harness_source_selection_ui",
                help="Choose which graph to query for codebase context."
            )
            st.session_state["harness_source_selection"] = "graphify" if "Graphify" in source_selection_label else "codegraph"
            
            retrieval_method_label = st.selectbox(
                "Retrieval Engine",
                ["Native/Internal CLI (with Python Fallback)", "Advanced Hybrid Scoring (Python)"],
                index=0,
                key="harness_retrieval_method_ui",
                help="Internal CLI uses native command line query engines. Advanced Hybrid uses global PageRank + BM25 search."
            )
            st.session_state["harness_retrieval_method"] = "advanced" if "Advanced" in retrieval_method_label else "internal"
            
        with c2:
            st.session_state["harness_max_nodes"] = st.slider(
                "Max Nodes (Context Budget)",
                min_value=2,
                max_value=30,
                value=8,
                key="harness_max_nodes_ui",
                help="Maximum number of node snippets to feed into LLM query context."
            )
            
            # Conditionally render fields
            if st.session_state["harness_source_selection"] == "graphify":
                graphify_mode_label = st.radio(
                    "Graphify Traversal Strategy",
                    ["Broad Architecture (BFS)", "Deep Execution Path (DFS)"],
                    index=0,
                    key="harness_graphify_mode_ui",
                    help="BFS explores components broadly. DFS follows execution sequences deeply."
                )
                st.session_state["harness_graphify_mode"] = "dfs" if "DFS" in graphify_mode_label else "bfs"
                st.session_state["harness_max_neighbors"] = 4 # default fallback
            else:
                st.session_state["harness_max_neighbors"] = st.slider(
                    "Max Neighbors (AST Hops)",
                    min_value=1,
                    max_value=15,
                    value=4,
                    key="harness_max_neighbors_ui",
                    help="Limit on neighboring connections retrieved from the active AST nodes."
                )

        # Rectify Mode toggle — full width below both columns
        st.markdown("---")
        rectify_on = st.toggle(
            "🔧 Rectify Mode (Error Correction)",
            value=st.session_state.get("harness_rectify_mode", False),
            key="harness_rectify_mode_ui",
            help="When enabled, the LLM is instructed to produce structured <code_fix> blocks. "
                 "These appear as interactive panels below each answer where you can Apply the fix "
                 "(auto-rebuilds CodeGraph & Graphify) and Download the corrected file."
        )
        st.session_state["harness_rectify_mode"] = rectify_on
        if rectify_on:
            st.info(
                "🚧 **Rectify Mode ON** — The LLM will now wrap proposed code fixes in `<code_fix>` XML. "
                "After each answer, look for the 🔧 **Proposed Code Fix** panel below the response."
            )
    
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

            # ── Code Fix Panels (assistant messages only) ──
            if msg["role"] == "assistant":
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

            # ── Per-Message Retrieval Inspector & Derived Context Details ──
            retrieval_recs = msg.get("retrieval_records", [])
            if retrieval_recs:
                with st.expander("🔍 Show Retrieval Inspector & Derived Context Details", expanded=False):
                    for rec_idx, rec in enumerate(retrieval_recs):
                        st.markdown(f"#### 🌐 Retrieval System: `{rec.get('type', 'Graph')}`")
                        
                        # System Badges
                        st.markdown(
                            f"• **Source System**: `{rec.get('source_system', 'N/A')}`  \n"
                            f"• **Retrieval Engine**: `{rec.get('retrieval_method', 'N/A')}`  \n"
                            f"• **Strategy**: `{rec.get('retrieval_strategy', 'N/A')}`"
                        )
                        
                        # Parameters if available
                        params_list = []
                        if rec.get("max_nodes"):
                            params_list.append(f"Max Nodes: **{rec['max_nodes']}**")
                        if rec.get("max_neighbors"):
                            params_list.append(f"AST Hops: **{rec['max_neighbors']}**")
                        if rec.get("graphify_mode"):
                            params_list.append(f"Traversal: **{rec['graphify_mode'].upper()}**")
                        if rec.get("rectify_mode"):
                            params_list.append("Rectify Mode: **ON**")
                        if params_list:
                            st.caption(" | ".join(params_list))
                            
                        # Derived Context / Subgraph Content
                        if rec.get("source_system") == "langgraph" or "LangGraph" in rec.get("type", ""):
                            st.markdown("##### 🧩 Derived Louvain Communities & Summaries")
                            per_comm = rec.get("per_comm_details", [])
                            if per_comm:
                                for item in per_comm:
                                    st.info(
                                        f"**Community {item.get('cid')}** (Relevance Score: {item.get('score', 0.0):.3f})\n\n"
                                        f"**Summary**: {item.get('summary', '')}\n\n"
                                        f"**Intermediate Answer**: {item.get('partial_answer', '')}"
                                    )
                            st.markdown("##### 📝 Merged Prompt Context")
                            st.code(rec.get("merged_context_prompt", ""), language="text")
                        else:
                            st.markdown("##### 🌲 Derived AST Codebase Subgraph")
                            nodes = rec.get("selected_nodes", [])
                            if nodes:
                                st.markdown("**Selected Graph Nodes:**")
                                for n in nodes:
                                    st.write(f"- `{n.get('node_id')}` [{n.get('type')}] `{n.get('label')}` ({n.get('file_path')}:{n.get('line_start')})")
                            st.markdown("**Source Snippets & Context Derived:**")
                            st.code(rec.get("context_retrieved", "No explicit context retrieved."), language="python")
                            
                        if rec_idx < len(retrieval_recs) - 1:
                            st.markdown("---")

            if "harness_history" in msg and msg["harness_history"]:
                with st.expander("🛠️ Show Master Loop Perception-Action Observations", expanded=False):
                    for idx, step in enumerate(msg["harness_history"]):
                        st.markdown(f"#### 🔄 Iteration {idx+1}")
                        st.write(f"**Thought:** {step.get('thought')}")
                        st.write(f"**Action:** Call `{step.get('tool')}` with `{step.get('tool_input')}`")
                        st.code(step.get('observation'), language="text")

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
                result = harness.execute(user_query, max_iterations=16, callback=update_ui)
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
            c_tokens = sum(r.get("context_tokens", 0) for r in recs)
            
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

            single_opt = p_tokens + c_tokens + r_tokens
            single_base = asset_baseline_single + p_tokens if asset_baseline_single > 0 else max(100, p_tokens * 10)
            savings = (1.0 - (single_opt / single_base)) * 100.0 if single_base > 0 else 0.0

            total_chat_opt_tokens += single_opt
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
                "Prompt Tokens": cd["p_tokens"],
                "Context Tokens": cd["c_tokens"],
                "Response Tokens": cd["r_tokens"],
                "Optimized Total": cd["single_opt"],
                "Raw Baseline": cd["single_base"],
                "Savings %": f"{cd['savings']:.1f}%"
            })

        import pandas as pd
        df_chats = pd.DataFrame(table_rows)
        st.dataframe(df_chats, use_container_width=True)

        # Expandable Detail Cards
        for cd in chat_details:
            with st.expander(f"💬 Chat #{cd['turn']}: `{cd['query'][:60]}` ({cd['sys_used']})", expanded=False):
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("Prompt Tokens", f"{cd['p_tokens']:,}")
                cc2.metric("Context Tokens", f"{cd['c_tokens']:,}")
                cc3.metric("Response Tokens", f"{cd['r_tokens']:,}")
                cc4.metric("Chat Savings", f"{cd['savings']:.1f}%")
                st.caption(f"**Single-Turn Baseline**: {cd['single_base']:,} tokens  |  **Optimized Total**: {cd['single_opt']:,} tokens")

    st.markdown("---")

    # ── Cumulative Total Comparison ──
    st.markdown("### 📈 Cumulative Multi-Turn Comparison & Savings")
    num_chats = len(chat_details)
    
    if num_chats == 0:
        cum_baseline = asset_baseline_single if asset_baseline_single > 0 else 1000
        cum_optimized = overhead_tokens
        cum_savings_pct = 0.0
    else:
        # Baseline doubles/multiplies across N chat turns
        cum_baseline = (num_chats * asset_baseline_single) + total_prompt_tokens
        cum_optimized = overhead_tokens + total_chat_opt_tokens
        cum_savings_pct = (1.0 - (cum_optimized / cum_baseline)) * 100.0 if cum_baseline > 0 else 0.0

    col_base, col_opt, col_pct = st.columns(3)
    with col_base:
        st.metric("Total Raw Baseline (Multiplied)", f"{cum_baseline:,} tokens")
        st.caption(f"Sending full raw Codebase ({codebase_tokens:,}) + PDF ({pdf_tokens:,}) for each of {max(1, num_chats)} chat turn(s).")
    with col_opt:
        st.metric("Total Optimized (Graph-Based)", f"{cum_optimized:,} tokens")
        st.caption("Graph Overhead + Prompt + Derived Contexts + Responses across all chats.")
    with col_pct:
        st.metric("Net Token Reduction", f"{cum_savings_pct:.1f}%")
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
