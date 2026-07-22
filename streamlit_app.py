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

    return storage, pipeline, repo_service, chat_service, llm_provider

storage, pipeline, repo_service, chat_service, base_llm_provider = services()

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
        st.subheader("📂 Ingest Codebase Graph (Phase 1)")
        repo_upload = st.file_uploader("Upload Python codebase (.zip)", type=["zip"])
        if repo_upload:
            with st.spinner("Extracting and building Tree-sitter CodeGraph..."):
                try:
                    repo_meta = ingest_uploaded_zip(repo_upload)
                    st.session_state.repo_id = repo_meta.repo_id
                    st.success(f"Successfully built CodeGraph for repository: {repo_meta.name}")
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
    
    col_code, col_pdf = st.columns(2)
    
    with col_code:
        st.markdown("### 🌲 Tree-sitter CodeGraph Network")
        repo_id = st.session_state.get("repo_id")
        if not repo_id:
            st.warning("Upload a python codebase in Tab 1 to visualize the CodeGraph.")
        else:
            graph_doc = storage.load_codegraph(repo_id)
            if not graph_doc:
                st.error("Could not load CodeGraph file.")
            else:
                with st.spinner("Compiling CodeGraph visualization..."):
                    # Map GraphDocument to NetworkX Graph
                    G_code = nx.Graph()
                    for node in graph_doc.nodes:
                        if node.node_type == "cli_output":
                            continue
                        G_code.add_node(
                            node.node_id,
                            label=node.label,
                            type=node.node_type,
                            community_id=0,
                            description=node.source_snippet or f"Line {node.line_start} in {node.file_path}"
                        )
                    for edge in graph_doc.edges:
                        G_code.add_edge(
                            edge.source_node,
                            edge.target_node,
                            relation_type=edge.edge_type
                        )
                    
                    try:
                        code_html = graph_to_pyvis(G_code, height="500px")
                        import streamlit.components.v1 as components
                        components.html(code_html, height=520, scrolling=False)
                    except Exception as e:
                        st.error(f"Could not render CodeGraph: {e}")
                        
    with col_pdf:
        st.markdown("### 🕸️ LangGraph Unstructured Community Network")
        G_pdf = st.session_state.get("unstructured_graph")
        if G_pdf is None:
            st.warning("Upload documents and build the LangGraph in Tab 1 to visualize the Network.")
        else:
            with st.spinner("Compiling LangGraph visualization..."):
                try:
                    pdf_html = graph_to_pyvis(G_pdf, height="500px")
                    import streamlit.components.v1 as components
                    components.html(pdf_html, height=520, scrolling=False)
                except Exception as e:
                    st.error(f"Could not render LangGraph: {e}")

# --- Tab 2: Master Loop QA ---
with main_tabs[2]:
    st.subheader("💬 stateless Master Loop QA")
    
    # 🔍 Retrieval Inspector (Graph Context & Prompts)
    with st.expander("🔍 Retrieval Inspector (Graph Context & Prompts)", expanded=False):
        if not st.session_state.get("retrieval_history"):
            st.info("No queries executed yet. Submit a query to inspect the exact retrieved graph subgraphs, community summaries, and prompt context.")
        else:
            latest = st.session_state["retrieval_history"][0]
            st.markdown(f"**Last Query**: `{latest['query']}` (Type: `{latest['type']}` at {latest['timestamp']})")
            
            if latest["type"] == "LangGraph (PDF)":
                st.markdown("### 🧩 Retrieved Louvain Communities & Partial Answers")
                for item in latest["per_comm_details"]:
                    st.markdown(f"#### 🌐 Community ID: {item['cid']} (Score: {item['score']:.3f})")
                    st.write(f"**Anchors**: {item['anchors']}")
                    st.write(f"**Community Summary**:")
                    st.info(item["summary"])
                    st.write(f"**Intermediate Partial Answer**:")
                    st.success(item["partial_answer"])
                
                st.markdown("### 📝 Merged Prompt Context (Sent to LLM)")
                st.code(latest["merged_context_prompt"], language="text")
                
            elif latest["type"] == "CodeGraph (AST)":
                st.markdown("### 🌲 Retrieved AST Codebase Subgraph & Definitions")
                st.code(latest["context_retrieved"], language="python")
                st.write("**Synthesized Answer**:")
                st.success(latest["answer"])
    
    # Render chat history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
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

        with st.spinner("Initializing central agentic harness loop..."):
            harness = AgentHarness(chat_service=chat_service, llm_provider=active_llm)
            try:
                result = harness.execute(user_query, max_iterations=16, callback=update_ui)
                final_answer = result["final_answer"]
                history_log = result["history"]
            except Exception as e:
                final_answer = f"Harness loop crashed with error: {str(e)}"
                history_log = [{"thought": "Harness execution failed", "tool": "none", "tool_input": "{}", "observation": str(e)}]
                
        # Render final answer
        with st.chat_message("assistant"):
            st.markdown(final_answer)
            with st.expander("🛠️ Final Perception-Action Observations", expanded=False):
                for idx, step in enumerate(history_log):
                    st.markdown(f"#### 🔄 Iteration {idx+1}")
                    st.write(f"**Thought:** {step.get('thought')}")
                    st.write(f"**Action:** Call `{step.get('tool')}` with `{step.get('tool_input')}`")
                    st.code(step.get('observation'), language="text")

        # Save to chat history
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": final_answer,
            "harness_history": history_log
        })
        st.session_state["harness_todo"] = st.session_state.get("harness_todo", [])
        st.rerun()

# --- Tab 3: Token Analytics ---
with main_tabs[3]:
    st.subheader("📊 Token Analytics & Context Savings")
    
    # Calculate baseline
    docs_list = st.session_state.get("unstructured_docs", [])
    total_chars = sum(d.get("char_count", 0) for d in docs_list)
    baseline_tokens = int(total_chars / 4)
    
    # Calculate optimized
    history = st.session_state.get("retrieval_history", [])
    latest_query = None
    optimized_tokens = 0
    
    if history:
        latest = history[0]
        latest_query = latest.get("query")
        if latest["type"] == "LangGraph (PDF)":
            optimized_tokens = int(len(latest.get("merged_context_prompt", "")) / 4)
        elif latest["type"] == "CodeGraph (AST)":
            optimized_tokens = int(len(latest.get("context_retrieved", "")) / 4)
            
    st.markdown("### 📈 Cost & Token Optimization Comparison")
    
    if baseline_tokens == 0:
        st.warning("Please upload guidelines / PDFs in Tab 1 to populate baseline metrics.")
    else:
        col_base, col_opt, col_pct = st.columns(3)
        with col_base:
            st.metric("Baseline Model Context (Full PDF)", f"{baseline_tokens:,} tokens")
            st.caption("Sending the entire raw PDF content to the LLM.")
        with col_opt:
            st.metric("Optimized Model Context (Graph-based)", f"{optimized_tokens:,} tokens")
            st.caption("Sending only relevant sub-graphs & community summaries.")
        with col_pct:
            savings_pct = 100.0 if optimized_tokens == 0 else (100.0 - (optimized_tokens / baseline_tokens * 100.0))
            st.metric("Context Size Reduction", f"{savings_pct:.1f}%")
            st.caption("Optimization efficiency compared to baseline.")
            
        # Draw Bar Chart Comparison
        import pandas as pd
        chart_df = pd.DataFrame({
            "Context Model": ["Baseline Model (Full PDF)", "Optimized Model (Graph-Based)"],
            "Tokens": [baseline_tokens, optimized_tokens]
        })
        st.bar_chart(chart_df, x="Context Model", y="Tokens", color="#6366f1")
        
        # Financial and performance implications
        st.markdown("#### 💡 Performance & Financial Highlights")
        # Estimate Cost Savings (e.g. Gemini 2.5 Flash input price: $0.075 / 1M tokens)
        baseline_cost = (baseline_tokens / 1_000_000) * 0.075
        opt_cost = (optimized_tokens / 1_000_000) * 0.075
        cost_diff = baseline_cost - opt_cost
        
        st.write(f"- **Financial Efficiency**: The baseline model costs around **${baseline_cost:.6f}** per query, whereas our optimized model costs **${opt_cost:.6f}**. You are saving **${cost_diff:.6f}** per call!")
        st.write("- **Rate Limit Safety**: By filtering out up to 95% of irrelevant context, the graph-optimized model significantly reduces the risk of hitting API token limit exhaustion (`429 Rate Limit Exceeded`).")
        st.write("- **Inference Latency**: Smaller, focused prompt contexts result in significantly lower time-to-first-token and faster synthesis response speeds.")
