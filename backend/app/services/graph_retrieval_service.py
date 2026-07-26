from __future__ import annotations

import re
import math
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass

from app.models.schemas import GraphEdge, GraphNode, SourceSnippet, TokenMeasurement
from app.services.storage import LocalStorage
from app.services.token_service import TokenService
from app.services.graphify_service import GraphifyService
from app.services.codegraph_service import CodeGraphService


@dataclass
class GraphRetrievalResult:
    context: str
    snippets: list[SourceSnippet]
    selected_nodes: list[GraphNode]
    selected_edges: list[GraphEdge]
    token_measurement: TokenMeasurement
    retrieval_strategy: str = "unknown"


class GraphRetrievalService:
    _download_attempted = False

    def __init__(self, storage: LocalStorage, token_service: TokenService) -> None:
        self.storage = storage
        self.token_service = token_service
        self.graphify_service = GraphifyService(storage=storage)
        self.codegraph_service = CodeGraphService()

    def _ensure_node_22(self) -> str:
        """Ensure Node.js 22+ with node:sqlite and FTS5 support is available. Returns node path or 'node'."""
        import sys
        import os
        import shutil
        import urllib.request
        import tarfile
        import ssl

        # Check if system node is available, >= 22.5.0, and supports node:sqlite with FTS5
        try:
            check_script = (
                "const db = new (require('node:sqlite').DatabaseSync)(':memory:'); "
                "db.exec('CREATE VIRTUAL TABLE t USING fts5(c);');"
            )
            proc = subprocess.run(["node", "--experimental-sqlite", "-e", check_script], capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                return "node"
        except Exception:
            pass

        # We only download the precompiled binary on Linux (Streamlit Cloud runs on Linux x64)
        if not sys.platform.startswith("linux"):
            return "node"

        node_dir = self.storage.data_dir / "bin" / "node-codegraph"
        node_bin = node_dir / "node"

        if node_bin.exists():
            # Verify that this binary is actually executable, >= 22.5.0, and supports node:sqlite with FTS5
            try:
                check_script = (
                    "const db = new (require('node:sqlite').DatabaseSync)(':memory:'); "
                    "db.exec('CREATE VIRTUAL TABLE t USING fts5(c);');"
                )
                proc = subprocess.run([str(node_bin), "--experimental-sqlite", "-e", check_script], capture_output=True, text=True, timeout=5)
                if proc.returncode == 0:
                    return str(node_bin)
            except Exception:
                pass

            # If it's invalid or doesn't support node:sqlite/FTS5, delete it so we can re-download (only if we haven't already tried this session)
            if not GraphRetrievalService._download_attempted:
                try:
                    if node_dir.exists():
                        shutil.rmtree(node_dir)
                except Exception:
                    pass

        # If we already tried downloading this session and it's still invalid, don't download again to prevent infinite loops
        if GraphRetrievalService._download_attempted:
            return "node"

        GraphRetrievalService._download_attempted = True

        # Download tarball (using CodeGraph self-contained release which vendors a custom Node binary compiled with FTS5)
        node_dir.parent.mkdir(parents=True, exist_ok=True)
        tar_path = self.storage.data_dir / "bin" / "node.tar.gz"
        url = "https://github.com/colbymchenry/codegraph/releases/download/v0.9.9/codegraph-linux-x64.tar.gz"

        try:
            # Bypass SSL certificate check (commonly required on Streamlit containers with outdated certs)
            ssl_context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=ssl_context, timeout=60) as response, tar_path.open("wb") as out_file:
                shutil.copyfileobj(response, out_file)
            
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=node_dir.parent)

            extracted_folder = node_dir.parent / "codegraph-linux-x64"
            if extracted_folder.exists():
                if node_dir.exists():
                    shutil.rmtree(node_dir)
                extracted_folder.rename(node_dir)

            if node_bin.exists():
                os.chmod(node_bin, 0o755)

            if tar_path.exists():
                tar_path.unlink()

            return str(node_bin)
        except Exception as exc:
            if tar_path.exists():
                try:
                    tar_path.unlink()
                except Exception:
                    pass
            # Propagate the error so the user gets notified exactly why the Node.js 22 download failed
            raise RuntimeError(
                f"Internal engine fails: Failed to auto-install Node.js 22 dependency on Streamlit Cloud. "
                f"Download URL: {url}. Error detail: {exc}."
            )

    def _compute_pagerank(self, nodes: list[GraphNode], edges: list[GraphEdge], damping: float = 0.85, max_iter: int = 20) -> dict[str, float]:
        n = len(nodes)
        if n == 0:
            return {}
        
        # Initialize PageRank equally
        pr = {node.node_id: 1.0 / n for node in nodes}
        
        # Build adjacency and incoming mappings
        out_degree: dict[str, int] = {}
        incoming: dict[str, list[str]] = {node.node_id: [] for node in nodes}
        
        for edge in edges:
            src, tgt = edge.source_node, edge.target_node
            if src in pr and tgt in pr:
                incoming[tgt].append(src)
                out_degree[src] = out_degree.get(src, 0) + 1
                
        # Power iteration
        for _ in range(max_iter):
            new_pr = {}
            # Redistribute sink rank (0 out-degree) equally
            sink_sum = sum(pr[node_id] for node_id, deg in out_degree.items() if deg == 0)
            
            for node in nodes:
                nid = node.node_id
                rank = (1.0 - damping) / n
                rank += damping * (sink_sum / n)
                
                for src in incoming[nid]:
                    rank += damping * (pr[src] / out_degree[src])
                    
                new_pr[nid] = rank
            pr = new_pr
            
        return pr

    def _read_node_code(self, repo_id: str, node: GraphNode, file_cache: dict[str, list[str]], max_chars: int | None = None) -> str:
        try:
            if node.node_type == "cli_output" or not node.file_path:
                snippet = node.source_snippet or ""
                if max_chars is not None and len(snippet) > max_chars:
                    return snippet[:max_chars] + "\n... [Snippet Truncated] ..."
                return snippet
            
            import os
            repo_root = self.storage.repo_source_dir(repo_id)
            abs_path = os.path.normpath(str(repo_root / node.file_path))
            
            if abs_path not in file_cache:
                try:
                    from app.services.file_utils import read_text_lossy
                    text = read_text_lossy(Path(abs_path))
                    file_cache[abs_path] = text.splitlines()
                except Exception:
                    file_cache[abs_path] = []
            
            lines = file_cache[abs_path]
            if not lines:
                snippet = node.source_snippet or ""
                if max_chars is not None and len(snippet) > max_chars:
                    return snippet[:max_chars] + "\n... [Snippet Truncated] ..."
                return snippet
            
            # line_start and line_end are 1-based indices
            start = (node.line_start or 1) - 1
            end = (node.line_end or node.line_start or 1)
            
            # Ensure indices are within bounds
            start = max(0, min(start, len(lines)))
            end = max(0, min(end, len(lines)))
            
            if start >= end:
                return ""
                
            code = "\n".join(lines[start:end])
            if max_chars is not None and len(code) > max_chars:
                return code[:max_chars] + "\n... [Snippet Truncated] ..."
            return code
        except Exception:
            snippet = node.source_snippet or ""
            if max_chars is not None and len(snippet) > max_chars:
                return snippet[:max_chars] + "\n... [Snippet Truncated] ..."
            return snippet

    def _extract_node_signature(self, node: GraphNode, code: str) -> str:
        # Signature is lightweight: node name, type, and the first few lines of its code (if present)
        parts = [node.label or "", node.node_type or ""]
        if code:
            lines = code.splitlines()[:5]
            parts.extend(lines)
        return "\n".join(parts)

    def _format_neighbor_snippet(self, text: str, limit_chars: int) -> str:
        if not text:
            return ""
        if len(text) <= limit_chars:
            return text
        
        prefix = text[:limit_chars]
        remaining = text[limit_chars:]
        extra_lines = []
        
        for line in remaining.splitlines():
            line_strip = line.strip()
            # Catch function/method definitions and class headers
            is_declaration = (
                line_strip.startswith("def ") or 
                line_strip.startswith("class ") or 
                line_strip.startswith("function ") or
                line_strip.startswith("async ") or 
                "export " in line_strip or 
                "interface " in line_strip or
                line_strip.startswith("public ") or
                line_strip.startswith("private ")
            )
            # Catch docstring headers and comment structures
            is_docstring = (
                line_strip.startswith('"""') or 
                line_strip.startswith("'''") or 
                line_strip.startswith("//") or 
                line_strip.startswith("/*") or 
                line_strip.startswith("*") or
                line_strip.startswith("#")
            )
            if is_declaration or is_docstring:
                extra_lines.append(line)
                
        if extra_lines:
            return prefix + "\n\n... [Snippet Truncated - Declarations & Headers Expanded] ...\n" + "\n".join(extra_lines)
        return prefix + "\n\n... [Snippet Truncated] ..."

    def build_context(
        self, repo_id: str, query: str, max_nodes: int = 8,
        source_selection: str = "codegraph", retrieval_method: str = "internal",
        graphify_mode: str = "bfs", max_anchors: int | None = None,
        max_neighbors: int | None = None,
    ) -> GraphRetrievalResult:
        codegraph = self.storage.load_codegraph(repo_id)
        graphify = self.storage.load_graphify(repo_id)
        if codegraph is None:
            raise ValueError("CodeGraph output not found for repo.")

        if source_selection == "hybrid":
            res_cg = self.build_context(
                repo_id=repo_id, query=query, max_nodes=max_nodes,
                source_selection="codegraph", retrieval_method=retrieval_method,
                graphify_mode=graphify_mode, max_anchors=max_anchors, max_neighbors=max_neighbors
            )
            if graphify:
                res_gf = self.build_context(
                    repo_id=repo_id, query=query, max_nodes=max_nodes,
                    source_selection="graphify", retrieval_method=retrieval_method,
                    graphify_mode=graphify_mode, max_anchors=max_anchors, max_neighbors=max_neighbors
                )
                merged_nodes = res_cg.selected_nodes + [n for n in res_gf.selected_nodes if n.node_id not in {x.node_id for x in res_cg.selected_nodes}]
                merged_edges = res_cg.selected_edges + [e for e in res_gf.selected_edges if (e.source_node, e.target_node) not in {(x.source_node, x.target_node) for x in res_cg.selected_edges}]
                merged_snippets = res_cg.snippets + [s for s in res_gf.snippets if s.file_path not in {x.file_path for x in res_cg.snippets}]
                
                merged_context = (
                    "=== CODEGRAPH (AST-level details) ===\n" + res_cg.context + "\n\n"
                    "=== GRAPHIFY (High-level Architecture) ===\n" + res_gf.context
                )
                return GraphRetrievalResult(
                    context=merged_context,
                    snippets=merged_snippets,
                    selected_nodes=merged_nodes,
                    selected_edges=merged_edges,
                    token_measurement=res_cg.token_measurement,
                    retrieval_strategy=f"Hybrid Codebase: CodeGraph + Graphify ({retrieval_method.upper()})"
                )
            else:
                return res_cg

        # ── Internal Graph Retrieval ──
        # Uses the native query engines of CodeGraph / Graphify (CLI first, Python fallback)
        if retrieval_method == "internal":
            return self._internal_retrieval(repo_id, query, max_nodes, source_selection, codegraph, graphify, graphify_mode=graphify_mode)

        # ── Advanced Hybrid Scoring ──
        # Delegates entirely to the dedicated method which handles all scoring (BM25 + PageRank + EdgeRank + LineMatch)
        return self._advanced_hybrid_retrieval(
            repo_id=repo_id,
            query=query,
            max_nodes=max_nodes,
            source_selection=source_selection,
            codegraph=codegraph,
            graphify=graphify,
            max_anchors=max_anchors,
            max_neighbors=max_neighbors
        )

    def _internal_retrieval(
        self, repo_id: str, query: str, max_nodes: int,
        source_selection: str, codegraph, graphify,
        graphify_mode: str = "bfs",
    ) -> GraphRetrievalResult:
        """Route to the native query engine of CodeGraph / Graphify based on source_selection.

        Only uses the external CLI tools (graphify CLI or CodeGraph Node.js).
        If the CLI fails, the exception is raised directly to the user so they can inspect why it failed.
        """
        if source_selection == "graphify":
            if not graphify or not graphify.nodes:
                raise ValueError("Graphify output is not available for this repository.")
            selected_nodes, selected_edges, _ = self._query_graphify(repo_id, graphify, query, max_nodes, graphify_mode=graphify_mode)
            gf_budget = max_nodes * 250
            mode_label = graphify_mode.upper()
            strategy = f"Internal Graph Retrieval (Graphify CLI | {mode_label} | Budget: {gf_budget})"
        else:  # "codegraph"
            selected_nodes, selected_edges, _ = self._query_codegraph(repo_id, codegraph, query, max_nodes)
            strategy = "Internal Graph Retrieval (CodeGraph CLI)"

        snippets = self._snippets(repo_id, selected_nodes)
        context = self._format_context(selected_nodes, selected_edges, snippets, retrieval_method="internal")
        measurement = self.token_service.measure_estimated("codegraph_graphify_optimized_context", context)

        return GraphRetrievalResult(
            context=context,
            snippets=snippets,
            selected_nodes=selected_nodes,
            selected_edges=selected_edges,
            token_measurement=measurement,
            retrieval_strategy=strategy,
        )

    def _query_graphify(self, repo_id: str, graphify, query: str, max_nodes: int, graphify_mode: str = "bfs") -> tuple[list[GraphNode], list[GraphEdge], bool]:
        """Try external Graphify CLI. Raise an error if it is missing or fails."""
        repo_root = self.storage.repo_source_dir(repo_id)
        target_budget = max_nodes * 250

        # Ensure standard graph.json exists under repo_root/graphify-out/graph.json in NetworkX node-link format
        out_dir = repo_root / "graphify-out"
        out_dir.mkdir(parents=True, exist_ok=True)
        graph_json_path = out_dir / "graph.json"

        formatted_nodes = []
        for node in graphify.nodes:
            formatted_nodes.append({
                "id": node.node_id,
                "label": node.label,
                "type": node.node_type,
                "file_path": node.file_path,
                "line_start": node.line_start,
                "line_end": node.line_end,
                "source_snippet": node.source_snippet,
                "metadata": node.metadata
            })
        formatted_links = []
        for edge in graphify.edges:
            formatted_links.append({
                "source": edge.source_node,
                "target": edge.target_node,
                "type": edge.edge_type,
                "score": edge.score,
                "metadata": edge.metadata
            })

        graph_json_path.write_text(
            json.dumps({"nodes": formatted_nodes, "links": formatted_links}, indent=2),
            encoding="utf-8"
        )

        # Dynamically locate the graphify executable (especially on Streamlit Cloud)
        import sys
        import shutil
        graphify_cli = shutil.which("graphify")
        if not graphify_cli:
            scripts_dir = Path(sys.executable).resolve().parent
            for suffix in [".exe", ".cmd", ""]:
                candidate = scripts_dir / f"graphify{suffix}"
                if candidate.exists():
                    graphify_cli = str(candidate)
                    break
            if not graphify_cli:
                graphify_cli = "graphify"

        cmd = [graphify_cli, "query", query, "--budget", str(target_budget)]
        if str(graphify_mode).lower() == "dfs":
            cmd.append("--dfs")

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except Exception as e:
            raise RuntimeError(
                f"Internal engine fails: Graphify CLI execution failed because the tool could not be found or executed. "
                f"Error detail: {e}. Please ensure the 'graphify' CLI is installed and in your PATH."
            )

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip() or (proc.stdout or "").strip()
            raise RuntimeError(
                f"Internal engine fails: Graphify CLI query command failed (Exit Code {proc.returncode}). "
                f"Output/Error: {stderr}"
            )
            
        cli_output = (proc.stdout or "").strip()
        if not cli_output:
            raise RuntimeError(
                "Internal engine fails: Graphify CLI query succeeded but returned empty output context."
            )
            
        cli_node = GraphNode(
            node_id="graphify:cli_result",
            node_type="cli_output",
            label="Graphify CLI Query Result",
            source_snippet=cli_output,
            metadata={"source": "graphify_cli", "query": query},
        )
        return [cli_node], [], True

    def _query_codegraph(self, repo_id: str, codegraph, query: str, max_nodes: int) -> tuple[list[GraphNode], list[GraphEdge], bool]:
        """Try external CodeGraph Node.js CLI. Raise an error if it is missing or fails."""
        repo_root = self.storage.repo_source_dir(repo_id)

        # Get node executable path (guarantees Node 22.5+ on Streamlit Cloud)
        node_path = self._ensure_node_22()

        # Ensure node_modules dependencies are installed (specifically for Streamlit Cloud startup/deployment)
        node_modules_dir = Path.cwd() / "node_modules" / "@colbymchenry" / "codegraph"
        if not node_modules_dir.exists():
            # If we downloaded a custom Node 22 path, we must use the npm binary in that same package or system npm
            npm_path = "npm"
            if node_path != "node":
                npm_cand = Path(node_path).parent / "npm"
                if npm_cand.exists():
                    npm_path = str(npm_cand)
            try:
                subprocess.run(
                    [npm_path, "install"],
                    cwd=str(Path.cwd()),
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                    check=True
                )
            except Exception as e:
                raise RuntimeError(
                    f"Internal engine fails: CodeGraph Node dependencies (npm install) could not be installed. "
                    f"Error detail: {e}. Please ensure Node.js and npm are installed and available."
                )

        script = (
            "(async () => {\n"
            "  try {\n"
            "    let CodeGraph;\n"
            "    try {\n"
            "      CodeGraph = require('@colbymchenry/codegraph').CodeGraph;\n"
            "    } catch (_) {\n"
            "      const mainRoot = process.env.MAIN_PROJECT_ROOT || process.cwd();\n"
            "      CodeGraph = require(require.resolve('@colbymchenry/codegraph', { paths: [mainRoot, process.cwd()] })).CodeGraph;\n"
            "    }\n"
            "    const projectRoot = process.cwd();\n"
            "    let cg;\n"
            "    if (CodeGraph.isInitialized(projectRoot)) {\n"
            "      cg = await CodeGraph.open(projectRoot);\n"
            "    } else {\n"
            "      cg = await CodeGraph.init(projectRoot);\n"
            "    }\n"
            "    if (cg.indexAll) await cg.indexAll();\n"
            "    const query = process.argv[2] || '';\n"
            f"    const ctx = await cg.buildContext(query, {{ maxNodes: {max_nodes}, includeCode: true, format: 'markdown' }});\n"
            "    console.log(JSON.stringify({{ context: ctx }}));\n"
            "    if (cg.close) await cg.close();\n"
            "  } catch (e) {\n"
            "    console.error(e && e.stack ? e.stack : e);\n"
            "    console.error('Executed Node.js version:', process.version);\n"
            "    process.exitCode = 2;\n"
            "  }\n"
            "})();\n"
        )
        script_path = repo_root / ".codegraph_query.js"
        script_path.write_text(script, encoding="utf-8")
        
        import os
        env = dict(os.environ)
        env["MAIN_PROJECT_ROOT"] = str(Path.cwd())
        
        try:
            proc = subprocess.run(
                [node_path, "--experimental-sqlite", str(script_path), query],
                cwd=str(repo_root),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
        except Exception as e:
            try:
                script_path.unlink()
            except Exception:
                pass
            raise RuntimeError(
                f"Internal engine fails: CodeGraph CLI execution failed because Node.js could not be found or executed. "
                f"Error detail: {e}. Please ensure 'node' (Node.js) is installed and in your PATH."
            )

        try:
            script_path.unlink()
        except Exception:
            pass

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip() or (proc.stdout or "").strip()
            node_version = "unknown"
            try:
                check_proc = subprocess.run([node_path, "-v"], capture_output=True, text=True, timeout=5)
                node_version = check_proc.stdout.strip()
            except Exception:
                pass
            raise RuntimeError(
                f"Internal engine fails: CodeGraph CLI Node.js process failed (Exit Code {proc.returncode}). "
                f"Executed Node version: {node_version} (Path: {node_path}). "
                f"Error: {stderr}"
            )

        stdout = (proc.stdout or "").strip()
        if not stdout:
            raise RuntimeError(
                "Internal engine fails: CodeGraph CLI query succeeded but returned empty output context."
            )

        try:
            parsed = json.loads(stdout)
            cli_context = parsed.get("context") or stdout
        except json.JSONDecodeError:
            cli_context = stdout

        cli_node = GraphNode(
            node_id="codegraph:cli_result",
            node_type="cli_output",
            label="CodeGraph CLI Query Result",
            source_snippet=cli_context,
            metadata={"source": "codegraph_cli", "query": query},
        )
        return [cli_node], [], True


    # ── Utility methods ────────────────────────────────────────────────────

    def _terms(self, query: str) -> list[str]:
        return [term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", query)]

    def _snippets(self, repo_id: str, nodes: list[GraphNode]) -> list[SourceSnippet]:
        """Extract source snippets from selected nodes (works for both CLI output and Python query nodes)."""
        snippets: list[SourceSnippet] = []
        seen: set[tuple[str, int | None, int | None]] = set()
        file_cache: dict[str, list[str]] = {}
        for node in nodes:
            code = self._read_node_code(repo_id, node, file_cache)
            if not code:
                continue
            # Use file_path if available, otherwise use node_id as identifier
            path = node.file_path or node.node_id
            key = (path, node.line_start, node.line_end)
            if key in seen:
                continue
            seen.add(key)
            snippets.append(
                SourceSnippet(
                    file_path=path,
                    line_start=node.line_start or 0,
                    line_end=node.line_end or node.line_start or 0,
                    text=code,
                    source="graph",
                )
            )
        return snippets

    def _format_context(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        snippets: list[SourceSnippet],
        retrieval_method: str = "advanced",
    ) -> str:
        if retrieval_method == "internal":
            mode_header = "### Retrieval Mode: Internal CLI Engine (CodeGraph/Graphify)"
        else:
            mode_header = "### Retrieval Mode: Advanced Hybrid Engine (BM25 + PageRank + EdgeRank + LineMatch)"

        node_lines = [
            f"- {node.node_id} [{node.node_type}] {node.label} ({node.file_path or 'external'}:{node.line_start or '-'})"
            for node in nodes
        ]
        edge_lines = [
            f"- {edge.source_node} --{edge.edge_type}--> {edge.target_node}"
            for edge in edges
        ]
        snippet_lines = [
            f"### {snippet.file_path}:{snippet.line_start}-{snippet.line_end} ({snippet.source})\n{snippet.text}"
            for snippet in snippets
        ]
        return "\n".join(
            [
                mode_header,
                "",
                "Graph-selected nodes:",
                *node_lines,
                "",
                "Graph relationships:",
                *edge_lines,
                "",
                "Source snippets:",
                *snippet_lines,
            ]
        )

    def _format_advanced_context(
        self,
        anchors: list[GraphNode],
        neighbors: list[GraphNode],
        connections: list[dict[str, str]],
        snippets: list[SourceSnippet]
    ) -> str:
        lines = [
            "### Retrieval Mode: Advanced Hybrid Engine (BM25 + PageRank + EdgeRank + LineMatch)",
            "",
            "Selected Anchor Nodes (ranked by query relevance + centrality):",
        ]
        for node in anchors:
            lines.append(f"- [Anchor] {node.node_id} ({node.file_path or 'external'}:{node.line_start or '-'})")
            
        lines.append("")
        lines.append("Selected Neighborhood Nodes:")
        for node in neighbors:
            lines.append(f"- [Neighbor] {node.node_id} ({node.file_path or 'external'}:{node.line_start or '-'})")
            
        lines.append("")
        lines.append("Graph Connections between Anchors and Neighbors:")
        for conn in connections:
            lines.append(f"- Anchor '{conn['anchor']}' is connected to Neighbor '{conn['neighbor']}' via relationship '{conn['edge_type']}'")
            
        lines.append("")
        lines.append("Source Code Snippets:")
        for snippet in snippets:
            lines.append(f"### File: {snippet.file_path} (Lines {snippet.line_start}-{snippet.line_end}) [{snippet.source}]")
            lines.append(snippet.text)
            lines.append("---")
            
        return "\n".join(lines)

    def _advanced_hybrid_retrieval(
        self, repo_id: str, query: str, max_nodes: int,
        source_selection: str, codegraph, graphify,
        max_anchors: int | None = None, max_neighbors: int | None = None,
    ) -> GraphRetrievalResult:
        # Source Selection and Dynamic Graph Loading
        if source_selection == "graphify" and graphify:
            all_nodes = graphify.nodes
            all_edges = graphify.edges
        else: # "codegraph"
            all_nodes = codegraph.nodes
            all_edges = codegraph.edges

        # 1. Base limits mapping
        if max_nodes <= 8:
            base_anchors, base_neighbors = 2, 4
        elif max_nodes <= 14:
            base_anchors, base_neighbors = 4, 8
        else:
            base_anchors, base_neighbors = 8, 16

        # Dynamically scale actual limits based on total nodes (N) in the active repository graph
        N = len(all_nodes)
        if N <= 100:
            scale_factor = 0.5
        elif N <= 1000:
            scale_factor = 1.0
        elif N <= 5000:
            scale_factor = 2.0
        else:
            scale_factor = 3.0

        if max_anchors is not None:
            actual_max_anchors = max_anchors
        else:
            actual_max_anchors = max(2, int(base_anchors * scale_factor))

        if max_neighbors is not None:
            actual_max_neighbors = max_neighbors
        else:
            actual_max_neighbors = max(4, int(base_neighbors * scale_factor))

        # 2. Run global PageRank centrality scoring (independent of query)
        pr_map = self._compute_pagerank(all_nodes, all_edges)
        sorted_by_pr = sorted(all_nodes, key=lambda n: pr_map.get(n.node_id, 0.0), reverse=True)
        pr_ranks = {node.node_id: idx + 1 for idx, node in enumerate(sorted_by_pr)}
        
        pr_scores = {}
        for node in all_nodes:
            rank = pr_ranks[node.node_id]
            pr_scores[node.node_id] = 10.0 * (1.0 - (rank - 1) / N) if N > 1 else 10.0

        # Read every node's code snippet to build BM25 corpus (truncate to 2000 chars to save memory/CPU)
        file_cache: dict[str, list[str]] = {}
        node_codes: dict[str, str] = {}
        for node in all_nodes:
            node_codes[node.node_id] = self._read_node_code(repo_id, node, file_cache, max_chars=2000)

        # 3. BM25 scoring (query-dependent)
        query_terms = [t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query) if len(t) > 1]
        
        # Calculate document frequencies for terms and fast document lengths
        doc_freqs = {term: 0 for term in query_terms}
        corpus_tokens = {}
        doc_lens = {}
        total_len = 0
        
        for node in all_nodes:
            nid = node.node_id
            code = node_codes[nid]
            if not code:
                doc_lens[nid] = 0
                continue
            
            # Fast document length estimation
            d_len = len(code.split())
            doc_lens[nid] = d_len
            total_len += d_len
            
            # Normalize tokens
            tokens = set(t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code))
            corpus_tokens[nid] = tokens
            for term in query_terms:
                if term in tokens:
                    doc_freqs[term] += 1
                    
        avg_doc_len = (total_len / N) if N > 0 else 1.0
        
        # Calculate TF-IDF / BM25
        k1 = 1.2
        b = 0.75
        bm25_scores = {}
        
        import math
        for node in all_nodes:
            nid = node.node_id
            score = 0.0
            tokens = corpus_tokens.get(nid, set())
            d_len = doc_lens.get(nid, 0)
            
            for term in query_terms:
                if term in tokens:
                    # BM25 IDF
                    df = doc_freqs[term]
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                    
                    # Term frequency inside document
                    code = node_codes[nid]
                    tf = code.lower().count(term)
                    
                    # Score contribution
                    tf_scaled = (tf * (k1 + 1)) / (tf + k1 * (1.0 - b + b * (d_len / avg_doc_len)))
                    score += idf * tf_scaled
                    
            bm25_scores[nid] = score

        # 4. Synthesize final scores (Linear Combination)
        w_bm25 = 0.65
        w_pr = 0.35
        node_scores = {}
        for node in all_nodes:
            nid = node.node_id
            # Normalize BM25 score
            raw_bm25 = bm25_scores.get(nid, 0.0)
            node_scores[nid] = (w_bm25 * raw_bm25) + (w_pr * pr_scores.get(nid, 0.0))

        # Sort and select top anchor nodes with Term-Balanced Multi-Anchor Selection
        anchors_list = sorted(all_nodes, key=lambda n: node_scores.get(n.node_id, 0.0), reverse=True)
        selected_anchors = {}
        
        # Ensure at least 1 top anchor per distinct query term is selected for cross-component queries
        if len(query_terms) > 1:
            for term in query_terms:
                term_matches = [
                    node for node in anchors_list 
                    if term in node.label.lower() or term in (node.file_path or "").lower()
                ]
                if term_matches:
                    best_match = term_matches[0]
                    selected_anchors[best_match.node_id] = best_match
                    if len(selected_anchors) >= actual_max_anchors:
                        break
                        
        # Fill remaining anchor slots from global score ranking
        for node in anchors_list:
            if len(selected_anchors) >= actual_max_anchors:
                break
            if node.node_id not in selected_anchors:
                selected_anchors[node.node_id] = node

        # 5. Graph Neighborhood Expansion (EdgeRank)
        # Select neighboring nodes directly connected to selected anchors
        adjacency: dict[str, list[tuple[str, str, float]]] = {}
        for edge in all_edges:
            src = edge.source_node
            tgt = edge.target_node
            score = edge.score or 0.5
            adjacency.setdefault(src, []).append((tgt, edge.edge_type, score))
            adjacency.setdefault(tgt, []).append((src, edge.edge_type, score))

        neighbor_candidates = {}
        connections_info = []
        for anchor_id in selected_anchors:
            for neighbor_id, rel_type, rel_score in adjacency.get(anchor_id, []):
                if neighbor_id in selected_anchors:
                    # Log connection details between anchors
                    connections_info.append({
                        "anchor": anchor_id,
                        "neighbor": neighbor_id,
                        "edge_type": rel_type
                    })
                    continue
                if neighbor_id not in neighbor_candidates:
                    # Find candidate node
                    candidate = next((n for n in all_nodes if n.node_id == neighbor_id), None)
                    if candidate:
                        neighbor_candidates[neighbor_id] = {
                            "node": candidate,
                            "rel_score": rel_score,
                            "rel_type": rel_type,
                            "anchor": selected_anchors[anchor_id].label
                        }
                else:
                    # Keep connection with highest score
                    if rel_score > neighbor_candidates[neighbor_id]["rel_score"]:
                        neighbor_candidates[neighbor_id]["rel_score"] = rel_score
                        neighbor_candidates[neighbor_id]["rel_type"] = rel_type
                        neighbor_candidates[neighbor_id]["anchor"] = selected_anchors[anchor_id].label

        # Sort and select top neighbors
        neighbors_list = sorted(
            neighbor_candidates.values(),
            key=lambda x: (x["rel_score"], node_scores.get(x["node"].node_id, 0.0)),
            reverse=True
        )
        selected_neighbors = {}
        for item in neighbors_list[:actual_max_neighbors]:
            node = item["node"]
            selected_neighbors[node.node_id] = node
            connections_info.append({
                "anchor": item["anchor"],
                "neighbor": node.label,
                "edge_type": item["rel_type"]
            })

        # 6. Synthesize context matching token budget
        selected_nodes = list(selected_anchors.values()) + list(selected_neighbors.values())
        
        # Extract matching edges
        selected_node_ids = set(selected_anchors.keys()) | set(selected_neighbors.keys())
        selected_edges = []
        for edge in all_edges:
            if edge.source_node in selected_node_ids and edge.target_node in selected_node_ids:
                selected_edges.append(edge)

        # Gather source code snippets
        seen_snippets = set()
        snippets = []
        for node in selected_nodes:
            code = node_codes.get(node.node_id)
            if code is None:
                code = self._read_node_code(repo_id, node, file_cache)
            node.source_snippet = code
            
            if not node.file_path and node.node_type != "cli_output":
                continue
            path = node.file_path or node.node_id
            key = (path, node.line_start, node.line_end)
            if key not in seen_snippets:
                seen_snippets.add(key)
                role = "anchor" if node.node_id in selected_anchors else "neighbor"
                snippets.append(
                    SourceSnippet(
                        file_path=path,
                        line_start=node.line_start or 1,
                        line_end=node.line_end or node.line_start or 1,
                        text=code,
                        source=f"graph_{role}",
                    )
                )

        context = self._format_advanced_context(
            anchors=list(selected_anchors.values()),
            neighbors=list(selected_neighbors.values()),
            connections=connections_info,
            snippets=snippets
        )
        measurement = self.token_service.measure_estimated("codegraph_graphify_optimized_context", context)
        
        return GraphRetrievalResult(
            context=context,
            snippets=snippets,
            selected_nodes=selected_nodes,
            selected_edges=selected_edges,
            token_measurement=measurement,
            retrieval_strategy="Advanced Hybrid Scoring System (BM25 + PageRank + EdgeRank + LineMatch)",
        )

