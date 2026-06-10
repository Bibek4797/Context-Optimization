from __future__ import annotations

import os
import shutil
import sys
import textwrap
import re
import difflib
import hashlib
from pathlib import Path
from uuid import uuid4

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.schemas import GraphDocument, QueryRecord, RepoMetadata, TreeNode  # noqa: E402
from app.services.analysis_pipeline import AnalysisPipeline  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.codegraph_service import CodeGraphService  # noqa: E402
from app.services.file_utils import clean_repo_name, safe_extract_zip  # noqa: E402
from app.services.graph_retrieval_service import GraphRetrievalService  # noqa: E402
from app.services.graphify_service import GraphifyService  # noqa: E402
from app.services.llm.gemini import GeminiProvider  # noqa: E402
from app.services.repo_service import RepoService  # noqa: E402
from app.services.storage import LocalStorage  # noqa: E402
from app.services.token_service import TokenService  # noqa: E402
from app.services.tree_sitter_service import TreeSitterService  # noqa: E402


st.set_page_config(
    page_title="Context Optimization Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
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


def ensure_node_dependencies() -> None:
    """Ensure that the Node.js package @colbymchenry/codegraph is installed."""
    import subprocess
    import shutil
    
    if not shutil.which("node"):
        return
        
    try:
        # Check if require('@colbymchenry/codegraph') works
        proc = subprocess.run(
            ["node", "-e", "require('@colbymchenry/codegraph')"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        if proc.returncode != 0:
            # Try with the --experimental-sqlite flag in case Node 22 requires it
            proc = subprocess.run(
                ["node", "--experimental-sqlite", "-e", "require('@colbymchenry/codegraph')"],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
        if proc.returncode == 0:
            return
    except Exception:
        return

    # If not installed, run npm install
    if shutil.which("npm"):
        try:
            subprocess.run(
                ["npm", "install", "--no-save", "@colbymchenry/codegraph"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                timeout=60
            )
        except Exception:
            pass


@st.cache_resource
def services():
    ensure_node_dependencies()
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
        api_key=get_secret("GEMINI_API_KEY"),
        model=get_secret("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash",
    )
    chat_service = ChatService(
        storage=storage,
        graph_retrieval_service=GraphRetrievalService(storage=storage, token_service=token_service),
        token_service=token_service,
        llm_provider=llm_provider,
        pipeline=pipeline,
    )

    return storage, pipeline, repo_service, chat_service, llm_provider


storage, pipeline, repo_service, chat_service, llm_provider = services()


# ── Streamlit Performance Cache Layers ──

@st.cache_resource(show_spinner=False)
def cached_load_codegraph(repo_id: str, updated_at: str) -> GraphDocument | None:
    graph = storage.load_codegraph(repo_id)
    if graph:
        from app.services.graph_retrieval_service import GraphRetrievalService
        from app.core.dependencies import token_service
        retrieval_service = GraphRetrievalService(storage, token_service)
        file_cache = {}
        for node in graph.nodes:
            node.source_snippet = retrieval_service._read_node_code(repo_id, node, file_cache)
    return graph

@st.cache_resource(show_spinner=False)
def cached_load_graphify(repo_id: str, updated_at: str) -> GraphDocument | None:
    graph = storage.load_graphify(repo_id)
    if graph:
        from app.services.graph_retrieval_service import GraphRetrievalService
        from app.core.dependencies import token_service
        retrieval_service = GraphRetrievalService(storage, token_service)
        file_cache = {}
        for node in graph.nodes:
            node.source_snippet = retrieval_service._read_node_code(repo_id, node, file_cache)
    return graph

@st.cache_data(show_spinner=False)
def cached_load_files_df(repo_id: str, updated_at: str) -> list[dict]:
    files = storage.load_files(repo_id)
    return [file.model_dump() for file in files]

@st.cache_data(show_spinner=False)
def cached_get_graph_dot(repo_id: str, kind: str, updated_at: str, max_nodes: int = 80, max_edges: int = 160) -> str:
    graph = cached_load_codegraph(repo_id, updated_at) if kind == "codegraph" else cached_load_graphify(repo_id, updated_at)
    if not graph:
        return ""
    return graph_to_dot(graph, max_nodes=max_nodes, max_edges=max_edges)

@st.cache_data(show_spinner=False)
def cached_get_graph_nodes_df(repo_id: str, kind: str, updated_at: str) -> list[dict]:
    graph = cached_load_codegraph(repo_id, updated_at) if kind == "codegraph" else cached_load_graphify(repo_id, updated_at)
    if not graph:
        return []
    return [node.model_dump() for node in graph.nodes]

@st.cache_data(show_spinner=False)
def cached_get_graph_edges_df(repo_id: str, kind: str, updated_at: str) -> list[dict]:
    graph = cached_load_codegraph(repo_id, updated_at) if kind == "codegraph" else cached_load_graphify(repo_id, updated_at)
    if not graph:
        return []
    return [edge.model_dump() for edge in graph.edges]

@st.cache_data(show_spinner=False)
def cached_load_queries(repo_id: str, queries_dir_path: str, count: int) -> list:
    from app.models.schemas import QueryRecord
    import json
    queries_dir = Path(queries_dir_path)
    queries = []
    if queries_dir.exists():
        for file_path in queries_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    queries.append(QueryRecord.model_validate(data))
            except Exception:
                pass
    return sorted(queries, key=lambda q: q.created_at)


def current_repo() -> RepoMetadata | None:
    repo_id = st.session_state.get("repo_id")
    if not repo_id:
        return None
    return storage.load_repo_metadata(repo_id)


def set_repo(repo: RepoMetadata) -> None:
    st.session_state.repo_id = repo.repo_id
    st.session_state.selected_file = None


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
    """Ingest one or more individual code files (not zipped)."""
    repo_id = uuid4().hex
    # Name the repo after the first file or "uploaded_files"
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


def metric_row(repo: RepoMetadata) -> None:
    cols = st.columns(4)
    cols[0].metric("Total files", repo.stats.total_files)
    cols[1].metric("Python files", repo.stats.python_files)
    cols[2].metric("Total lines", repo.stats.total_lines)
    cols[3].metric("Python lines", repo.stats.python_lines)


def render_status(repo: RepoMetadata | None) -> None:
    model_info = llm_provider.get_model_info()
    st.sidebar.subheader("Status")
    st.sidebar.write(f"Repo: **{repo.name if repo else 'none'}**")
    st.sidebar.write(f"Pipeline: **{repo.status if repo else 'idle'}**")
    st.sidebar.write(f"Model: **{model_info.model}**")
    st.sidebar.write(f"Gemini key: **{'configured' if model_info.configured else 'missing'}**")
    
    st.sidebar.subheader("Codebase Rectifier")
    st.session_state.rectify_enabled = st.sidebar.checkbox(
        "🔍 Enable Error Rectification",
        value=False,
        help="When enabled, the assistant will propose direct code modifications when bugs or errors are identified, allowing you to overwrite files with a single click."
    )


def render_notice(message: str) -> None:
    if "Graphify" in message and "fallback" in message.lower():
        st.info(message)
        return
    st.warning(message)





def render_upload_import(repo: RepoMetadata | None) -> None:
    st.header("Upload Or Import")

    SUPPORTED_EXTENSIONS = [
        "py", "pyi", "js", "jsx", "ts", "tsx",
        "go", "rs", "java", "c", "cpp", "cc", "cxx", "h", "hpp",
    ]

    # ── Section 1: Upload individual code files ──
    st.subheader("📄 Upload Code Files")
    st.caption("Upload one or more individual source files directly — no zipping needed.")
    uploaded_files = st.file_uploader(
        "Choose source files",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        key="file_uploader",
    )
    if st.button("Analyze files", disabled=not uploaded_files, type="primary"):
        with st.spinner("Analyzing uploaded files..."):
            try:
                new_repo = ingest_uploaded_files(uploaded_files)
                set_repo(new_repo)
                st.success(f"Loaded {new_repo.name} ({len(uploaded_files)} file{'s' if len(uploaded_files) != 1 else ''})")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.markdown("---")

    # ── Section 2: Upload repo (zip or GitHub) ──
    st.subheader("📦 Upload Repository")
    left, right = st.columns(2)
    with left:
        st.markdown("**Upload zipped codebase**")
        uploaded_zip = st.file_uploader("Choose .zip file", type=["zip"], key="zip_uploader")
        if st.button("Analyze zip", disabled=uploaded_zip is None, type="primary"):
            with st.spinner("Extracting and analyzing repository..."):
                try:
                    new_repo = ingest_uploaded_zip(uploaded_zip)
                    set_repo(new_repo)
                    st.success(f"Loaded {new_repo.name}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    with right:
        st.markdown("**Import GitHub URL**")
        github_url = st.text_input("Repository URL", placeholder="https://github.com/owner/repo")
        if st.button("Clone and analyze", disabled=not github_url.strip()):
            with st.spinner("Cloning and analyzing repository..."):
                try:
                    new_repo = repo_service.import_github(github_url.strip())
                    set_repo(new_repo)
                    st.success(f"Loaded {new_repo.name}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    st.caption(
        "Supported languages: Python, JavaScript/TypeScript, Go, Rust, Java, and C/C++. "
        "The app automatically skips .git, virtual environments (.venv), node_modules, build outputs, and caches."
    )

    if repo:
        st.markdown("---")
        st.subheader("Repo Analysis")
        metric_row(repo)
        st.write(f"Origin: `{repo.origin}`")
        if repo.error:
            st.error(repo.error)
        if repo.warnings:
            with st.expander("⚠️ System Warnings", expanded=False):
                for warning in repo.warnings:
                    render_notice(warning)
        files_df = cached_load_files_df(repo.repo_id, str(repo.updated_at))
        clean_files = [
            {"File Path": f.get("path"), "Language": f.get("language")}
            for f in files_df
        ]
        st.subheader("Files Metadata")
        st.dataframe(clean_files, use_container_width=True, hide_index=True)


def node_label(node: TreeNode) -> str:
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    suffix = f" | {node.text_preview}" if node.text_preview else ""
    return f"{node.type} [{start_line}:{node.start_point[1]} - {end_line}:{node.end_point[1]}]{suffix}"


def collect_tree_rows(
    node: TreeNode,
    depth: int,
    max_depth: int,
    budget: list[int],
    rows: list[dict[str, str | int | bool | None]],
) -> None:
    if budget[0] <= 0:
        return
    budget[0] -= 1
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    rows.append(
        {
            "tree": f"{'  ' * depth}{node.type}",
            "depth": depth,
            "named": node.named,
            "start": f"{start_line}:{node.start_point[1]}",
            "end": f"{end_line}:{node.end_point[1]}",
            "preview": node.text_preview,
        }
    )
    if depth < max_depth:
        for child in node.children:
            collect_tree_rows(child, depth + 1, max_depth, budget, rows)


def flatten_named_nodes(node: TreeNode, output: list[TreeNode], limit: int = 500) -> None:
    if len(output) >= limit:
        return
    if node.named:
        output.append(node)
    for child in node.children:
        flatten_named_nodes(child, output, limit)


def dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', "'").replace("\n", "\\n")


def tree_to_dot(root: TreeNode, max_depth: int = 5, max_nodes: int = 180) -> tuple[str, bool]:
    lines = [
        "digraph TreeSitter {",
        "rankdir=TB;",
        'node [shape=box, style="rounded,filled", fillcolor="#fbfcfa", color="#bfc8c2", fontsize=10];',
        'edge [color="#6d4b7d"];',
    ]
    counter = {"value": 0}
    truncated = {"value": False}

    def visit(node: TreeNode, depth: int) -> str | None:
        if counter["value"] >= max_nodes:
            truncated["value"] = True
            return None
        current_id = f"n{counter['value']}"
        counter["value"] += 1
        start_line = node.start_point[0] + 1
        label = dot_escape(f"{node.type}\\nL{start_line}")
        fill = "#e6f4f1" if node.named else "#f7f8f6"
        lines.append(f'"{current_id}" [label="{label}", fillcolor="{fill}"];')
        if depth < max_depth:
            for child in node.children:
                child_id = visit(child, depth + 1)
                if child_id:
                    lines.append(f'"{current_id}" -> "{child_id}";')
        elif node.children:
            truncated["value"] = True
        return current_id

    visit(root, 0)
    lines.append("}")
    return "\n".join(lines), truncated["value"]


def render_tree_sitter(repo: RepoMetadata | None) -> None:
    st.header("Tree-sitter Explorer")
    if not repo:
        st.info("Load a repository first.")
        return
    files = storage.load_files(repo.repo_id)
    if not files:
        st.info("No Python files available.")
        return
    selected = st.selectbox(
        "File",
        [file.path for file in files],
        index=0,
        key="tree_file_select",
    )
    document = storage.load_tree_sitter(repo.repo_id, selected)
    if not document:
        st.error("Tree-sitter output not found.")
        return
    if document.warnings:
        for warning in document.warnings:
            st.warning(warning)
    if document.parse_error:
        st.error(document.parse_error)
        st.code(document.source, language="python")
        return

    left, right = st.columns([0.42, 0.58])
    with left:
        st.subheader("Parse Tree")
        max_depth = st.slider("Expansion depth", 2, 8, 5)
        max_nodes = st.slider("Rendered nodes", 50, 1000, 300, step=50)
        if document.root:
            rows: list[dict[str, str | int | bool | None]] = []
            collect_tree_rows(document.root, 0, max_depth, [max_nodes], rows)
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.subheader("Parse Tree Graph")
            dot, truncated = tree_to_dot(document.root, max_depth=max_depth, max_nodes=min(max_nodes, 220))
            if truncated:
                st.info("Graph view is truncated by the depth/node controls to keep the page responsive.")
            st.graphviz_chart(dot, use_container_width=True)
    with right:
        st.subheader("Source Span")
        if document.root:
            named_nodes: list[TreeNode] = []
            flatten_named_nodes(document.root, named_nodes)
            labels = [node_label(node) for node in named_nodes]
            selected_index = st.selectbox("Highlight node", range(len(labels)), format_func=lambda index: labels[index])
            chosen = named_nodes[selected_index]
            start_line = chosen.start_point[0] + 1
            end_line = chosen.end_point[0] + 1
            lines = document.source.splitlines()
            snippet = "\n".join(lines[max(0, start_line - 1) : min(len(lines), end_line)])
            st.caption(f"{selected}:{start_line}-{end_line}")
            st.code(snippet or document.source, language="python", line_numbers=True)


def graph_to_dot(graph: GraphDocument, max_nodes: int = 80, max_edges: int = 160) -> str:
    visible_nodes = graph.nodes[:max_nodes]
    visible = {node.node_id for node in visible_nodes}
    lines = [
        "digraph G {", 
        "rankdir=LR;", 
        "bgcolor=transparent;",
        'node [shape=box, style="rounded,filled", fontname="Courier New", fontsize=9];'
    ]
    for node in visible_nodes:
        ntype = (node.node_type or "").lower()
        label = f"[{ntype.upper()}]\\n{node.label}".replace('"', "'")
        node_id_escaped = node.node_id.replace('"', "'")
        
        # Harmonious color palette per node type
        if ntype in {"module", "file"}:
            fill, border = "#e7f5ff", "#228be6" # Blue
        elif ntype in {"class", "struct_item", "component"}:
            fill, border = "#ebfbee", "#40c057" # Green
        elif ntype in {"function", "method", "concept"}:
            fill, border = "#fff9db", "#fab005" # Yellow/Amber
        else:
            fill, border = "#f8f9fa", "#dee2e6" # Slate/Light Gray
            
        lines.append(f'"{node_id_escaped}" [label="{label}", fillcolor="{fill}", color="{border}", penwidth=1.5];')
        
    for edge in graph.edges:
        if edge.source_node in visible and edge.target_node in visible:
            label = edge.edge_type.replace('"', "'")
            lines.append(
                f'"{edge.source_node}" -> "{edge.target_node}" [label="{label}", color="#495057", fontcolor="#495057", fontsize=8];'
            )
            max_edges -= 1
            if max_edges <= 0:
                break
    lines.append("}")
    return "\n".join(lines)


def render_graph_schematic(kind: str) -> None:
    st.markdown("---")
    st.subheader("🔮 Semantic Schema Guide")
    st.markdown(
        "This interactive blueprint explains the **graph model schema** and **metadata layers** "
        "captured in your graph database. This metadata is selectively loaded to optimize LLM query contexts."
    )
    
    if kind == "codegraph":
        t1, t2 = st.tabs(["📝 AST Node Types & Metadata", "🔗 AST Relationship Edges"])
        with t1:
            st.markdown(
                """
                | Node Type | Represents | Captured Information & Metadata Keys | Purpose in optimized QA |
                | :--- | :--- | :--- | :--- |
                | **`module`** | A file in the codebase (e.g. `.py`, `.ts`). | `file_path`, `total_lines`, `docstring_headers`, `imports` | Serves as the high-level anchor for structural organization. |
                | **`class`** | An OOP class definition. | `label` (class name), base classes, `line_start`, `line_end` | Defines data structures and component boundaries. |
                | **`function`** / **`method`** | Independent utility functions or class-bound methods. | `signature`, parameters, return type, docstrings | Defines modular logic boundaries (code loaded on-demand from disk). |
                """
            )

        with t2:
            st.markdown(
                """
                | Edge Type | Connection | Meta Info Saved | Architectural Meaning |
                | :--- | :--- | :--- | :--- |
                | **`contains`** | `module ➔ class` or `class ➔ method` | Parent-child ownership, lexical nesting. | Map OOP structures and hierarchy. |
                | **`calls`** | `function ➔ function` or `method ➔ function` | Caller position, line number, frequency. | Traces dynamic execution paths and call graph. |
                | **`imports`** | `module ➔ module` | Imported entities, aliases, source line. | Maps compilation and module dependency chains. |
                | **`inherits`** | `class ➔ class` | Subclassing relations, base class names. | Identifies inheritance trees and behavioral overrides. |
                """
            )
    else: # graphify
        t1, t2 = st.tabs(["🏗️ Macro Design Nodes & Metadata", "🌊 Lifted Flow Edges"])
        with t1:
            st.markdown(
                """
                | Node Type | Represents | Captured Information & Metadata Keys | Purpose in optimized QA |
                | :--- | :--- | :--- | :--- |
                | **`file`** | A high-level module file. | `file_path`, dependency weights, import footprint | Maps structural macro organization. |
                | **`component`** | A macro-level class or concept design boundary. | Design role, interaction count, encapsulation level | Identifies core concepts and system boundaries. |
                """
            )

        with t2:
            st.markdown(
                """
                | Edge Type | Lifted Relation | Meta Info Saved | Architectural Meaning |
                | :--- | :--- | :--- | :--- |
                | **`part_of`** | `component ➔ file` | Structural containing, component ownership. | Maps component-to-file bundling. |
                | **`depends_on`** | `file ➔ file` | Compilation imports, import dependency trees. | Tracks high-level architectural layering. |
                | **`triggers`** | `component ➔ component` | Aggregated call flows, execution triggers. | Maps execution triggers across modular component bounds. |
                | **`extends`** | `component ➔ component` | Architectural inheritance, class expansions. | Tracks modular extension hierarchies. |
                """
            )
            
    with st.expander("🔬 View Raw JSON Database Schema Definitions"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Graph Node Schema Example:**")
            st.code(
                '''{
  "node_id": "codegraph:app/services/token_service.py:TokenService:estimate_tokens",
  "node_type": "method",
  "label": "estimate_tokens",
  "file_path": "backend/app/services/token_service.py",
  "line_start": 15,
  "line_end": 23,
  "source_snippet": null,
  "metadata": {
    "docstring_headers": ["Estimate tokens using tiktoken."],
    "parameters": ["self", "text"],
    "return_type": "int"
  }
}''',
                language="json"
            )
        with col2:
            st.markdown("**Graph Edge Schema Example:**")
            st.code(
                '''{
  "edge_id": "edge:token_service:calls:tiktoken",
  "edge_type": "calls",
  "source_node": "codegraph:app/services/token_service.py:TokenService:estimate_tokens",
  "target_node": "external:tiktoken:get_encoding",
  "score": 1.0,
  "metadata": {
    "line_number": 20,
    "alias": "tiktoken"
  }
}''',
                language="json"
            )


def generate_interactive_html_graph(graph, kind: str) -> str:
    import json
    nodes = []
    for node in graph.nodes:
        nodes.append({
            "id": node.node_id,
            "label": node.label,
            "type": node.node_type,
            "file_path": node.file_path,
            "source_snippet": node.source_snippet
        })
    links = []
    for edge in graph.edges:
        links.append({
            "source": edge.source_node,
            "target": edge.target_node,
            "type": edge.edge_type,
            "score": edge.score
        })
    
    data_json = json.dumps({"nodes": nodes, "links": links})
    
    title_str = "CodeGraph Explorer" if kind == "codegraph" else "Graphify Explorer"
    subtitle_str = "Full Repository Dependency and Call Hierarchy Graph" if kind == "codegraph" else "Macro-Level High-Level Architectural Flow Graph"
    
    template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Interactive {title_str}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
            overflow: hidden;
        }}
        #canvas {{
            width: 100vw;
            height: 100vh;
        }}
        .node {{
            stroke: #21262d;
            stroke-width: 2px;
            cursor: pointer;
            transition: r 0.2s, stroke-width 0.2s;
        }}
        .node:hover {{
            stroke: #8b949e;
            stroke-width: 3px;
        }}
        .node text {{
            fill: #c9d1d9;
            font-size: 11px;
            font-family: monospace;
            pointer-events: none;
            text-shadow: 0 1px 2px rgba(0,0,0,0.8);
        }}
        .link {{
            stroke: #30363d;
            stroke-opacity: 0.6;
            stroke-width: 1.5px;
            fill: none;
        }}
        #tooltip {{
            position: absolute;
            background: rgba(22, 27, 34, 0.95);
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px;
            color: #c9d1d9;
            font-size: 13px;
            pointer-events: none;
            box-shadow: 0 4px 20px rgba(0,0,0,0.6);
            display: none;
            font-family: monospace;
            z-index: 100;
        }}
        .header {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(22, 27, 34, 0.85);
            padding: 15px 20px;
            border-radius: 8px;
            border: 1px solid #30363d;
            backdrop-filter: blur(8px);
            pointer-events: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .header h3 {{ margin: 0 0 5px 0; color: #58a6ff; font-size: 18px; }}
        .header p {{ margin: 0; font-size: 12px; color: #8b949e; }}
        .controls {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(22, 27, 34, 0.85);
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #30363d;
            display: flex;
            flex-direction: column;
            gap: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        button {{
            background: #21262d;
            border: 1px solid #30363d;
            color: #c9d1d9;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: background 0.2s, border-color 0.2s;
        }}
        button:hover {{
            background: #30363d;
            border-color: #8b949e;
        }}
        .legend {{
            margin-top: 10px;
            display: flex;
            flex-direction: column;
            gap: 5px;
            font-size: 11px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 1px solid #fff;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h3>🌲 {title_str}</h3>
        <p>{subtitle_str}</p>
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background-color: #4cf07b"></div> File / Module</div>
            <div class="legend-item"><div class="legend-color" style="background-color: #58a6ff"></div> Class / Component</div>
            <div class="legend-item"><div class="legend-color" style="background-color: #ff7b72"></div> Function / Concept</div>
            <div class="legend-item"><div class="legend-color" style="background-color: #d2a8ff"></div> Method</div>
            <div class="legend-item"><div class="legend-color" style="background-color: #8b949e"></div> External / Other</div>
        </div>
    </div>
    <svg id="canvas"></svg>
    <div id="tooltip"></div>
    <div class="controls">
        <button id="reset-btn">Reset Zoom</button>
        <button id="pause-btn">Pause Physics</button>
    </div>

    <script>
        const graphData = {data_json};

        const svg = d3.select("#canvas");
        const width = window.innerWidth;
        const height = window.innerHeight;
        svg.attr("width", width).attr("height", height);

        const container = svg.append("g");

        // Zoom & Pan
        const zoom = d3.zoom()
            .scaleExtent([0.05, 10])
            .on("zoom", (event) => {{
                container.attr("transform", event.transform);
            }});
        svg.call(zoom);

        document.getElementById("reset-btn").addEventListener("click", () => {{
            svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
        }});

        // Color coding
        const colors = {{
            "file": "#4cf07b",
            "module": "#4cf07b",
            "component": "#58a6ff",
            "class": "#58a6ff",
            "concept": "#ff7b72",
            "function": "#ff7b72",
            "method": "#d2a8ff",
            "external": "#8b949e"
        }};
        const get_color = (type) => colors[type] || "#bc8cff";

        // Links
        const link = container.append("g")
            .selectAll("line")
            .data(graphData.links)
            .enter().append("line")
            .attr("class", "link");

        // Nodes Group
        const node = container.append("g")
            .selectAll(".node-group")
            .data(graphData.nodes)
            .enter().append("g")
            .attr("class", "node-group")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("circle")
            .attr("class", "node")
            .attr("r", d => d.type === "file" || d.type === "module" ? 10 : 8)
            .attr("fill", d => get_color(d.type));

        node.append("text")
            .attr("dx", 12)
            .attr("dy", ".35em")
            .text(d => d.label || d.id);

        // Force Simulation
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(120))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(25));

        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("transform", d => "translate(" + d.x + "," + d.y + ")");
        }});

        // Tooltip interaction
        const tooltip = d3.select("#tooltip");
        node.selectAll("circle")
            .on("mouseover", function(event, d) {{
                let tooltipContent = "<strong>ID:</strong> " + d.id + "<br/><strong>Type:</strong> " + d.type;
                if (d.file_path) {{
                    tooltipContent += "<br/><strong>File:</strong> " + d.file_path;
                }}
                if (d.source_snippet) {{
                    let escaped = d.source_snippet
                        .replace(/&/g, "&amp;")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;");
                    tooltipContent += "<br/><strong>Code Snippet:</strong><pre style='margin: 4px 0 0 0; background: #161b22; padding: 6px; border-radius: 4px; border: 1px solid #30363d; font-size: 10px; max-height: 120px; overflow-y: auto; white-space: pre-wrap; font-family: monospace; text-align: left;'>" + escaped + "</pre>";
                }}
                tooltip.style("display", "block").html(tooltipContent);
                d3.select(this).attr("r", d.type === "file" || d.type === "module" ? 14 : 12);
            }})
            .on("mousemove", function(event) {{
                tooltip.style("left", (event.pageX + 15) + "px")
                       .style("top", (event.pageY - 15) + "px");
            }})
            .on("mouseout", function(event, d) {{
                tooltip.style("display", "none");
                d3.select(this).attr("r", d.type === "file" || d.type === "module" ? 10 : 8);
            }});

        // Controls
        let physicsPaused = false;
        document.getElementById("pause-btn").addEventListener("click", () => {{
            if (physicsPaused) {{
                simulation.alpha(0.3).restart();
                document.getElementById("pause-btn").innerText = "Pause Physics";
            }} else {{
                simulation.stop();
                document.getElementById("pause-btn").innerText = "Resume Physics";
            }}
            physicsPaused = !physicsPaused;
        }});

        function dragstarted(event, d) {{
            if (!event.active && !physicsPaused) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active && !physicsPaused) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}

        window.addEventListener("resize", () => {{
            const w = window.innerWidth;
            const h = window.innerHeight;
            svg.attr("width", w).attr("height", h);
            if (!physicsPaused) simulation.force("center", d3.forceCenter(w / 2, h / 2)).restart();
        }});
    </script>
</body>
</html>"""
    return template


def render_graph(repo: RepoMetadata | None, kind: str) -> None:
    title = "CodeGraph Explorer" if kind == "codegraph" else "Graphify Explorer"
    st.header(title)
    if not repo:
        st.info("Load a repository first.")
        return
    graph = cached_load_codegraph(repo.repo_id, str(repo.updated_at)) if kind == "codegraph" else cached_load_graphify(repo.repo_id, str(repo.updated_at))
    if not graph:
        st.error(f"{title} output not found.")
        return
    if graph.warnings:
        for warning in graph.warnings:
            render_notice(warning)
    if kind == "graphify" and graph.source == "graphify-fallback":
        st.info(
            "Native Graphify output is not available in this environment. This tab is showing the saved Graphify adapter output plus a clearly labeled fallback graph derived from CodeGraph."
        )
    # Download Interactive HTML Graph option (Placed at the top)
    try:
        html_content = generate_interactive_html_graph(graph, kind)
        st.download_button(
            label=f"📥 Download Interactive {title} HTML",
            data=html_content,
            file_name=f"{kind}_graph_{repo.repo_id[:8]}.html",
            mime="text/html",
            key=f"download_html_{kind}"
        )
    except Exception as e:
        st.caption(f"Could not generate downloadable HTML graph: {e}")

    cols = st.columns(3)
    cols[0].metric("Source", graph.source)
    cols[1].metric("Nodes", len(graph.nodes))
    cols[2].metric("Edges", len(graph.edges))
    st.graphviz_chart(cached_get_graph_dot(repo.repo_id, kind, str(repo.updated_at)), use_container_width=True)

    with st.expander("Nodes"):
        st.dataframe(cached_get_graph_nodes_df(repo.repo_id, kind, str(repo.updated_at)), use_container_width=True, hide_index=True)
    with st.expander("Edges"):
        st.dataframe(cached_get_graph_edges_df(repo.repo_id, kind, str(repo.updated_at)), use_container_width=True, hide_index=True)
    if kind == "graphify" and graph.raw_output_path:
        with st.expander("Raw Graphify adapter output"):
            raw_path = Path(graph.raw_output_path)
            if raw_path.exists():
                raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
                st.code(raw_text[:16000], language="json")
                if len(raw_text) > 16000:
                    st.caption("Raw output truncated for display.")
            else:
                st.caption(f"Raw output path recorded but not found on disk: {graph.raw_output_path}")

    # Render semantic schema explanation view
    render_graph_schematic(kind)


def render_tokens(repo: RepoMetadata | None) -> None:
    st.header("Token Analytics")
    if not repo:
        st.info("Load a repository first.")
        return

    # 1. Load all queries run so far
    queries_dir = storage.repo_state_dir(repo.repo_id) / "queries"
    count = len(list(queries_dir.glob("*.json"))) if queries_dir.exists() else 0
    queries = cached_load_queries(repo.repo_id, str(queries_dir), count)

    if queries:
        st.subheader("📊 Query Token Usage & Savings History")
        
        # Build one unified table showing everything
        table_rows = []
        for q_rec in queries:
            if q_rec.mode != "graph_optimized":
                continue
            q_text = q_rec.query.strip()
            prompt_meas = q_rec.token_usage.get("llm_prompt_tokens")
            prompt_tokens = prompt_meas.tokens if prompt_meas else 0
            
            baseline_meas = q_rec.token_usage.get("whole_codebase_baseline")
            baseline_tokens = baseline_meas.tokens if baseline_meas else 0
            
            resp_meas = q_rec.token_usage.get("llm_response_tokens")
            resp_tokens = resp_meas.tokens if resp_meas else 0
            
            total_meas = q_rec.token_usage.get("total_per_query_tokens")
            total_tokens = total_meas.tokens if total_meas else (prompt_tokens + resp_tokens)
            
            saved = max(0, baseline_tokens - prompt_tokens) if baseline_tokens > 0 else 0
            pct = (saved / baseline_tokens * 100) if baseline_tokens > 0 else 0.0
            raw_strategy = getattr(q_rec, "retrieval_strategy", "") or ""
            if "Advanced" in raw_strategy:
                engine_name = "Advanced Hybrid"
            elif "Graphify" in raw_strategy:
                engine_name = "Graphify CLI"
            elif "CodeGraph" in raw_strategy:
                engine_name = "CodeGraph CLI"
            else:
                if q_rec.selected_nodes and any(n.node_id.startswith("graphify:") for n in q_rec.selected_nodes):
                    engine_name = "Graphify CLI"
                else:
                    engine_name = "CodeGraph CLI"

            table_rows.append({
                "Time": q_rec.created_at.strftime("%H:%M:%S") if q_rec.created_at else "N/A",
                "Query / Question": q_text,
                "Whole Codebase Baseline": baseline_tokens,
                "Graph-Optimized Input": prompt_tokens,
                "LLM Response Output": resp_tokens,
                "Total Query Tokens": total_tokens,
                "Tokens Saved": saved,
                "Savings %": f"{round(pct, 2)}%",
                "Engine / Strategy": engine_name
            })
            
        if table_rows:
            st.dataframe(table_rows, use_container_width=True, hide_index=True)
            
            # Show aggregate savings
            total_baseline = sum(q_rec.token_usage.get("whole_codebase_baseline").tokens for q_rec in queries if q_rec.mode == "graph_optimized" and q_rec.token_usage.get("whole_codebase_baseline"))
            total_graph = sum(q_rec.token_usage.get("llm_prompt_tokens").tokens for q_rec in queries if q_rec.mode == "graph_optimized" and q_rec.token_usage.get("llm_prompt_tokens"))
            total_saved = max(0, total_baseline - total_graph)
            avg_pct = (total_saved / total_baseline * 100) if total_baseline else 0
            
            st.markdown("#### 📈 Cumulative Savings Against Whole Codebase Baseline")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Baseline Tokens", f"{total_baseline:,}")
            c2.metric("Total Optimized Tokens", f"{total_graph:,}")
            c3.metric("Total Tokens Saved", f"{total_saved:,}")
            c4.metric("Avg. Token Savings %", f"{round(avg_pct, 2)}%")
        else:
            st.info("Ask a question in **Graph QA** to see direct query-by-query token savings here!")
    else:
        st.info("No queries have been run in this session yet. Go to Graph QA to ask a question!")

    # Collapsible Ingestion Pipeline Measurements
    summary = storage.load_token_summary(repo.repo_id)
    if summary:
        if summary.cumulative_session_usage:
            st.markdown("#### Cumulative Raw Stage Measurements")
            st.dataframe(
                [{"stage": key, "tokens": value} for key, value in summary.cumulative_session_usage.items()],
                use_container_width=True,
                hide_index=True,
            )


def parse_and_render_code_fix(answer_text: str, repo_id: str) -> None:
    if not answer_text:
        st.markdown("_No answer generated._")
        return
        
    # Regex to extract <code_fix>...</code_fix>
    match = re.search(r"<code_fix>(.*?)</code_fix>", answer_text, re.DOTALL)
    if not match:
        st.markdown(answer_text)
        return
        
    # Exclude the code_fix tags from the conversational output
    conversation_text = answer_text.replace(match.group(0), "").strip()
    if conversation_text:
        st.markdown(conversation_text)
        
    # Parse inner XML tags
    inner = match.group(1)
    file_match = re.search(r"<filepath>(.*?)</filepath>", inner, re.DOTALL)
    orig_match = re.search(r"<original_code>(.*?)</original_code>", inner, re.DOTALL)
    repl_match = re.search(r"<replacement_code>(.*?)</replacement_code>", inner, re.DOTALL)
    
    if not file_match or not orig_match or not repl_match:
        st.warning("⚠️ A code fix was proposed but could not be parsed completely.")
        return
        
    filepath = file_match.group(1).strip()
    original_code = orig_match.group(1)
    if original_code.startswith("\n"):
        original_code = original_code[1:]
    if original_code.endswith("\n"):
        original_code = original_code[:-1]
        
    replacement_code = repl_match.group(1)
    if replacement_code.startswith("\n"):
        replacement_code = replacement_code[1:]
    if replacement_code.endswith("\n"):
        replacement_code = replacement_code[:-1]
        
    st.markdown("---")
    st.markdown(f"#### 🛠️ Proposed Fix for `{filepath}`")
    
    # Compute unified diff
    orig_lines = original_code.splitlines()
    repl_lines = replacement_code.splitlines()
    diff_generator = difflib.unified_diff(
        orig_lines, 
        repl_lines, 
        fromfile=f"a/{filepath}", 
        tofile=f"b/{filepath}", 
        lineterm=""
    )
    diff_text = "\n".join(list(diff_generator))
    
    st.code(diff_text, language="diff", line_numbers=True)
    
    # Render Apply Fix button
    button_key = f"apply_fix_btn_{hashlib.md5(filepath.encode()).hexdigest()}"
    if st.button("📁 Apply Fix directly to Workspace", key=button_key, type="primary"):
        with st.spinner("Applying codebase changes and re-ingesting..."):
            res = chat_service.apply_rectification(
                repo_id=repo_id,
                file_path=filepath,
                original_code=original_code,
                replacement_code=replacement_code
            )
            if res.get("status") == "success":
                st.success(res.get("message"))
                st.balloons()
                st.info("🔄 Repository has been successfully re-indexed in the background! Ask another question to see the updated graph.")
            else:
                st.error(f"❌ Error applying fix: {res.get('error')}")





def render_query_record(record: QueryRecord) -> None:
    if record.error:
        st.error(record.error)
    strategy = getattr(record, "retrieval_strategy", "unknown")
    if strategy != "unknown":
        st.caption(f"{record.status.upper()} | {record.latency_ms} ms | Strategy: `{strategy}` | query_id={record.query_id}")
    else:
        st.caption(f"{record.status.upper()} | {record.latency_ms} ms | query_id={record.query_id}")
    parse_and_render_code_fix(record.answer, record.repo_id)

    
    # Calculate Token and Context Size Analytics
    baseline_prompt = record.token_usage.get("whole_codebase_baseline")
    optimized_prompt = record.token_usage.get("llm_prompt_tokens")
    response_tokens = record.token_usage.get("llm_response_tokens")
    
    baseline_count = baseline_prompt.tokens if baseline_prompt else 0
    optimized_count = optimized_prompt.tokens if optimized_prompt else 0
    response_count = response_tokens.tokens if response_tokens else 0
    
    saved_tokens = max(0, baseline_count - optimized_count)
    saved_percent = round((saved_tokens / baseline_count * 100), 1) if baseline_count else 0.0
    
    st.markdown("---")
    st.subheader("📊 Token & Context Size Savings Analysis")
    
    if baseline_count > 0:
        cols = st.columns(4)
        cols[0].metric(
            "General Baseline", 
            f"{baseline_count:,} tokens",
            help="Total prompt tokens required if you uploaded the entire codebase folder directly to the LLM."
        )
        cols[1].metric(
            "Optimized Graph", 
            f"{optimized_count:,} tokens",
            help="Actual prompt tokens sent using your graph-optimized retrieval."
        )
        cols[2].metric(
            "Net Input Saved", 
            f"{saved_tokens:,} tokens", 
            f"-{saved_percent}%" if saved_percent > 0 else "0.0%",
            help="Absolute and percentage reduction in LLM prompt (input) tokens."
        )
        cols[3].metric(
            "LLM Output Tokens", 
            f"{response_count:,} tokens",
            help="Actual response tokens generated by Gemini to answer your question."
        )
    else:
        st.info("Ingesting token summary... Ask a question to compute complete savings metrics!")
        
    with st.expander("🔍 Detailed Token Breakdown"):
        st.markdown(
            f"""
            * **General Chatbot Baseline (Whole Codebase):** `{baseline_count:,}` tokens
            * **Our Optimized Graph QA (Actual):** `{optimized_count:,}` tokens
            * **Net Input Tokens Saved:** `{saved_tokens:,}` tokens (**{saved_percent}% prompt size reduction**!)
            * **LLM Response Output size:** `{response_count:,}` tokens
            
            This means your graph-optimized retrieval algorithm saved **{saved_tokens:,} input tokens** on this single query!
            """
        )
        st.dataframe([value.model_dump() for value in record.token_usage.values()], use_container_width=True, hide_index=True)

    if record.selected_nodes:
        counts = graph_node_source_counts(record)
        cols = st.columns(2)
        cols[0].metric("CodeGraph nodes", counts["codegraph"])
        cols[1].metric("Native Graphify nodes", counts["graphify"])
        
        exact_context = getattr(record, "context", "") or ""
        
        with st.expander("📄 View Exact Context Passed to LLM", expanded=False):
            t1, t2 = st.tabs(["📝 Context Text Block Only", "🚀 Complete Raw LLM Prompt Ingested"])
            with t1:
                st.markdown(
                    "This is the exact, optimized context text block that was appended to the LLM prompt "
                    "using the selected graph retrieval engine:"
                )
                st.text_area("LLM Prompt Context Text", exact_context, height=350)
            with t2:
                st.markdown(
                    "This is the **entire raw prompt** received by the Gemini model, including "
                    "system role guidelines, automated code rectifier instructions, your question, and the context block:"
                )
                rectify_str = ""
                if st.session_state.get("rectify_enabled", False):
                    rectify_str = (
                        "\n\nIMPORTANT: If you identify any bug/error and propose a code change, you MUST wrap the proposed fix exactly in "
                        "the following XML structure so the system can apply it automatically:\n"
                        "<code_fix>\n"
                        "  <filepath>relative/path/to/file.py</filepath>\n"
                        "  <original_code>\n"
                        "// Exact block of old code to replace (must match precisely including spacing)\n"
                        "  </original_code>\n"
                        "  <replacement_code>\n"
                        "// Exact block of new code to insert\n"
                        "  </replacement_code>\n"
                        "</code_fix>\n"
                        "Make sure that the <original_code> block you target matches the codebase content exactly, character-for-character."
                    )
                full_raw_prompt = (
                    "You are a graph-aware repository QA assistant. Use the selected CodeGraph and Graphify nodes, "
                    "relationships, and snippets to answer. Prefer precise relationships over broad guesses. "
                    "Cite files, lines, and relevant graph nodes. If the graph context is insufficient, say so. "
                    "Be extremely concise, direct, and brief in your answer. Avoid verbose explanations."
                    f"{rectify_str}\n\n"
                    f"Question:\n{record.query}\n\n"
                    f"Optimized graph context:\n{exact_context}"
                )
                st.text_area("Gemini Final Prompt", full_raw_prompt, height=450)



        with st.expander("Selected Graph Nodes"):
            st.dataframe([node.model_dump() for node in record.selected_nodes], use_container_width=True, hide_index=True)
            
    with st.expander("Source Snippets", expanded=True):
        for snippet in record.source_snippets:
            st.caption(f"{snippet.file_path}:{snippet.line_start}-{snippet.line_end} | {snippet.source}")
            st.code(snippet.text, language="python", line_numbers=True)




def graph_node_source_counts(record: QueryRecord) -> dict[str, int]:
    counts = {"codegraph": 0, "graphify": 0}
    for node in record.selected_nodes:
        is_graphify = (
            node.node_id.startswith("graphify:") or 
            (node.metadata and node.metadata.get("graphify_processed") is True)
        )
        
        if is_graphify and not node.node_id.startswith("codegraph:"):
            counts["graphify"] += 1
        elif node.node_id.startswith("codegraph:"):
            counts["codegraph"] += 1
    return counts


def qa_prompt_help() -> str:
    return "Ask about architecture, important functions, call paths, classes, imports, or implementation behavior."





def render_codegraph_qa(repo: RepoMetadata | None) -> None:
    st.header("CodeGraph QA")
    if not repo:
        st.info("Load a repository first.")
        return
        
    codegraph_path = storage.repo_state_dir(repo.repo_id) / "codegraph.json"
    if not codegraph_path.exists() or codegraph_path.stat().st_size == 0:
        st.error("No CodeGraph graph found. Please build or import a repository with CodeGraph output first.")
        return

    left, right = st.columns([3, 1])
    with left:
        question = st.text_area(
            "Ask a question about codebase symbols and relationships:",
            placeholder="e.g. 'How does TokenService calculate prompt savings?'",
            key="codegraph_qa_question"
        )
    with right:
        retrieval_method_label = st.selectbox(
            "Retrieval Engine",
            ["Internal Graph Retrieval (Default)", "Advanced Hybrid Scoring System"],
            index=0,
            key="codegraph_retrieval_method",
            help="Select the retrieval algorithm for CodeGraph."
        )
        retrieval_method = "advanced" if "Advanced" in retrieval_method_label else "internal"

        max_nodes_input = st.text_input(
            "Max Nodes:",
            value="8",
            key="codegraph_max_nodes_input",
            disabled=(retrieval_method == "advanced"),
            help="Limits context details sent to the LLM. Larger number provides more files but uses more tokens."
        )
        
        max_anchors = None
        max_neighbors = None
        if retrieval_method == "advanced":
            max_anchors_input = st.text_input(
                "Primary Anchors (Full):",
                value="4",
                key="codegraph_max_anchors",
                help="Number of primary anchor nodes to include with full code."
            )
            max_neighbors_input = st.text_input(
                "Neighbors (Full):",
                value="8",
                key="codegraph_max_neighbors",
                help="Number of neighboring nodes to include with full code."
            )
            if max_anchors_input.strip():
                try:
                    max_anchors = int(max_anchors_input.strip())
                except ValueError:
                    st.warning("Please enter a valid integer for Primary Anchors.")
            if max_neighbors_input.strip():
                try:
                    max_neighbors = int(max_neighbors_input.strip())
                except ValueError:
                    st.warning("Please enter a valid integer for Neighbors.")

    max_nodes_val = 8
    if max_nodes_input.strip():
        try:
            max_nodes_val = int(max_nodes_input.strip())
        except ValueError:
            st.warning("Please enter a valid integer for Max Nodes.")
            
    if st.button("Ask CodeGraph QA", disabled=not question.strip(), type="primary"):
        with st.spinner("Selecting graph neighborhood and calling Gemini..."):
            record = chat_service.graph_optimized_qa(
                repo.repo_id,
                question.strip(),
                st.session_state.session_id,
                source_selection="codegraph",
                max_nodes=max_nodes_val,
                rectify=st.session_state.get("rectify_enabled", False),
                retrieval_method=retrieval_method,
                graphify_mode="bfs",
                max_anchors=max_anchors,
                max_neighbors=max_neighbors
            )
            st.session_state.codegraph_record = record

    if "codegraph_record" in st.session_state:
        render_query_record(st.session_state.codegraph_record)


def render_graphify_qa(repo: RepoMetadata | None) -> None:
    st.header("Graphify QA")
    if not repo:
        st.info("Load a repository first.")
        return
        
    graphify_path = storage.repo_state_dir(repo.repo_id) / "graphify.json"
    if not graphify_path.exists() or graphify_path.stat().st_size == 0:
        st.error("No Graphify graph found. Please build or import a repository with Graphify output first.")
        return

    left, right = st.columns([3, 1])
    with left:
        question = st.text_area(
            "Ask a question about codebase architecture:",
            placeholder="e.g. 'How is TokenService connected to standard QA?'",
            key="graphify_qa_question"
        )
    with right:
        traversal_label = st.radio(
            "Traversal Strategy",
            ["Broad Architecture (BFS)", "Deep Execution Path (DFS)"],
            index=0,
            key="graphify_traversal_mode",
            help="BFS explores the call graph broadly. DFS follows execution paths deeply."
        )
        graphify_mode = "dfs" if "DFS" in traversal_label else "bfs"
        
        retrieval_method_label = st.selectbox(
            "Retrieval Engine",
            ["Internal Graph Retrieval (Default)", "Advanced Hybrid Scoring System"],
            index=0,
            key="graphify_retrieval_method",
            help="Select the retrieval algorithm for Graphify."
        )
        retrieval_method = "advanced" if "Advanced" in retrieval_method_label else "internal"

        token_budget_input = st.text_input(
            "Max Token Usage (Budget):",
            value="2000",
            key="graphify_token_budget",
            disabled=(retrieval_method == "advanced"),
            help="Max characters/tokens budget for retrieval. Maps to max_nodes = max(1, budget // 250)."
        )
        
        max_anchors = None
        max_neighbors = None
        if retrieval_method == "advanced":
            max_anchors_input = st.text_input(
                "Primary Anchors (Full):",
                value="4",
                key="graphify_max_anchors",
                help="Number of primary anchor nodes to include with full code."
            )
            max_neighbors_input = st.text_input(
                "Neighbors (Full):",
                value="8",
                key="graphify_max_neighbors",
                help="Number of neighboring nodes to include with full code."
            )
            if max_anchors_input.strip():
                try:
                    max_anchors = int(max_anchors_input.strip())
                except ValueError:
                    st.warning("Please enter a valid integer for Primary Anchors.")
            if max_neighbors_input.strip():
                try:
                    max_neighbors = int(max_neighbors_input.strip())
                except ValueError:
                    st.warning("Please enter a valid integer for Neighbors.")

    token_budget_val = 2000
    if token_budget_input.strip():
        try:
            token_budget_val = int(token_budget_input.strip())
        except ValueError:
            st.warning("Please enter a valid integer for the token budget.")
            
    max_nodes = max(1, token_budget_val // 250)

    if st.button("Ask Graphify QA", disabled=not question.strip(), type="primary"):
        with st.spinner("Selecting graph neighborhood and calling Gemini..."):
            record = chat_service.graph_optimized_qa(
                repo.repo_id,
                question.strip(),
                st.session_state.session_id,
                source_selection="graphify",
                max_nodes=max_nodes,
                rectify=st.session_state.get("rectify_enabled", False),
                retrieval_method=retrieval_method,
                graphify_mode=graphify_mode,
                max_anchors=max_anchors,
                max_neighbors=max_neighbors
            )
            st.session_state.graphify_record = record

    if "graphify_record" in st.session_state:
        render_query_record(st.session_state.graphify_record)





def inject_premium_styles() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        
        <style>
            /* Global dark-mode background and font */
            html, body, [class*="css"], .stApp {
                font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: #0b0f19 !important;
                color: #f9fafb !important;
            }
            
            /* JetBrains Mono for Code Blocks */
            code, pre, [data-testid="stMarkdownContainer"] code {
                font-family: 'JetBrains Mono', source-code-pro, Menlo, Monaco, Consolas, "Courier New", monospace !important;
            }
            
            /* Professional Gradient Header */
            .main-header {
                font-size: 2.4rem;
                font-weight: 800;
                background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #3b82f6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.8rem;
                letter-spacing: -0.025em;
            }
            
            /* Glassmorphic card metrics */
            div[data-testid="stMetric"] {
                background: rgba(17, 24, 39, 0.6) !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 12px !important;
                padding: 15px 20px;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3) !important;
                backdrop-filter: blur(8px) !important;
                -webkit-backdrop-filter: blur(8px) !important;
                transition: all 0.25s ease-in-out;
            }
            
            div[data-testid="stMetric"]:hover {
                transform: translateY(-2px);
                border-color: rgba(99, 102, 241, 0.4);
                box-shadow: 0 10px 25px rgba(99, 102, 241, 0.15) !important;
            }
            
            /* Sidebar Styling */
            section[data-testid="stSidebar"] {
                background-color: #070a12 !important;
                border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
            }
            
            /* Tabs customization */
            button[data-baseweb="tab"] {
                font-size: 0.95rem !important;
                font-weight: 600 !important;
                color: #9ca3af !important;
                transition: all 0.2s ease !important;
                background-color: transparent !important;
                border: none !important;
                padding: 8px 16px !important;
            }
            button[data-baseweb="tab"]:hover {
                color: #f3f4f6 !important;
                background-color: rgba(255, 255, 255, 0.02) !important;
            }
            button[data-baseweb="tab"][aria-selected="true"] {
                color: #a5b4fc !important;
                background-color: rgba(99, 102, 241, 0.1) !important;
                border-radius: 6px;
            }
            
            /* Smooth transitions for interactive buttons */
            .stButton>button {
                border-radius: 8px !important;
                font-weight: 600 !important;
                background: rgba(255, 255, 255, 0.05) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                color: #f3f4f6 !important;
                transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }
            
            .stButton>button:hover {
                background: rgba(255, 255, 255, 0.08) !important;
                border-color: rgba(255, 255, 255, 0.2) !important;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
                transform: translateY(-1px) !important;
            }
            
            /* Primary buttons gradient */
            .stButton>button[data-testid="baseButton-primary"] {
                background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
                color: #ffffff !important;
                border: none !important;
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
            }
            .stButton>button[data-testid="baseButton-primary"]:hover {
                background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
                box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
                transform: translateY(-1px) !important;
            }
            
            /* Custom style for inputs */
            div[data-baseweb="select"] > div {
                background-color: #111827 !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 8px !important;
            }
            div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
                background-color: #111827 !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 8px !important;
                color: #f9fafb !important;
            }
            
            /* Premium Alert & Info Boxes */
            div[data-testid="stNotification"] {
                border-radius: 8px !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                background-color: rgba(17, 24, 39, 0.6) !important;
                backdrop-filter: blur(8px) !important;
            }
            
            /* Dataframes and tables */
            div[data-testid="stDataFrame"] {
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 12px !important;
                overflow: hidden !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_premium_styles()
    st.markdown('<h1 class="main-header">Context Optimization Engine</h1>', unsafe_allow_html=True)

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex

    repo = current_repo()
    render_status(repo)

    tab_names = [
        "Upload / Import",
        "CodeGraph",
        "CodeGraph QA",
        "Graphify",
        "Graphify QA",
        "Token Analytics",
    ]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        render_upload_import(repo)
    with tabs[1]:
        render_graph(repo, "codegraph")
    with tabs[2]:
        render_codegraph_qa(repo)
    with tabs[3]:
        render_graph(repo, "graphify")
    with tabs[4]:
        render_graphify_qa(repo)
    with tabs[5]:
        render_tokens(repo)




if __name__ == "__main__":
    main()
