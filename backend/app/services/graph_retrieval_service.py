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
    def __init__(self, storage: LocalStorage, token_service: TokenService) -> None:
        self.storage = storage
        self.token_service = token_service
        self.graphify_service = GraphifyService(storage=storage)
        self.codegraph_service = CodeGraphService()

    def _ensure_node_22(self) -> str:
        """Ensure Node.js 22+ is available on Linux/Streamlit. Returns node path or 'node'."""
        import sys
        import os
        import shutil
        import urllib.request
        import tarfile

        # Check if system node is >= 22.5.0 (node:sqlite support)
        try:
            proc = subprocess.run(["node", "-v"], capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                v_str = proc.stdout.strip().lstrip("v")
                parts = v_str.split(".")
                if parts:
                    major = int(parts[0])
                    minor = int(parts[1]) if len(parts) > 1 else 0
                    if major > 22 or (major == 22 and minor >= 5):
                        return "node"
        except Exception:
            pass

        # We only download the precompiled binary on Linux (Streamlit Cloud runs on Linux x64)
        if sys.platform != "linux":
            return "node"

        node_dir = self.storage.data_dir / "bin" / "node-v22.11.0"
        node_bin = node_dir / "bin" / "node"

        if node_bin.exists():
            return str(node_bin)

        # Download tarball
        node_dir.parent.mkdir(parents=True, exist_ok=True)
        tar_path = self.storage.data_dir / "bin" / "node.tar.xz"
        url = "https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz"

        try:
            with urllib.request.urlopen(url, timeout=30) as response, tar_path.open("wb") as out_file:
                shutil.copyfileobj(response, out_file)
            
            with tarfile.open(tar_path, "r:xz") as tar:
                tar.extractall(path=node_dir.parent)

            extracted_folder = node_dir.parent / "node-v22.11.0-linux-x64"
            if extracted_folder.exists():
                if node_dir.exists():
                    shutil.rmtree(node_dir)
                extracted_folder.rename(node_dir)

            if node_bin.exists():
                os.chmod(node_bin, 0o755)

            if tar_path.exists():
                tar_path.unlink()

            return str(node_bin)
        except Exception:
            if tar_path.exists():
                try:
                    tar_path.unlink()
                except Exception:
                    pass
            return "node"

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

    def _read_node_code(self, repo_id: str, node: GraphNode, file_cache: dict[str, list[str]]) -> str:
        if node.node_type == "cli_output" or not node.file_path:
            return node.source_snippet or ""
        
        repo_root = self.storage.repo_source_dir(repo_id)
        abs_path = str((repo_root / node.file_path).resolve())
        
        if abs_path not in file_cache:
            try:
                from app.services.file_utils import read_text_lossy
                text = read_text_lossy(Path(abs_path))
                file_cache[abs_path] = text.splitlines()
            except Exception:
                file_cache[abs_path] = []
        
        lines = file_cache[abs_path]
        if not lines:
            return node.source_snippet or ""
        
        # line_start and line_end are 1-based indices
        start = (node.line_start or 1) - 1
        end = (node.line_end or node.line_start or 1)
        
        # Ensure indices are within bounds
        start = max(0, min(start, len(lines)))
        end = max(0, min(end, len(lines)))
        
        if start >= end:
            return ""
            
        return "\n".join(lines[start:end])

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

    def build_context(self, repo_id: str, query: str, max_nodes: int = 8, source_selection: str = "codegraph", retrieval_method: str = "internal", graphify_mode: str = "bfs") -> GraphRetrievalResult:
        codegraph = self.storage.load_codegraph(repo_id)
        graphify = self.storage.load_graphify(repo_id)
        if codegraph is None:
            raise ValueError("CodeGraph output not found for repo.")

        # ── Internal Graph Retrieval ──
        # Uses the native query engines of CodeGraph / Graphify (CLI first, Python fallback)
        if retrieval_method == "internal":
            return self._internal_retrieval(repo_id, query, max_nodes, source_selection, codegraph, graphify, graphify_mode=graphify_mode)

        # ── Advanced Hybrid Scoring (needs structural node/edge lists) ──
        # Source Selection and Dynamic Graph Loading
        if source_selection == "graphify" and graphify:
            all_nodes = graphify.nodes
            all_edges = graphify.edges
        else: # "codegraph"
            all_nodes = codegraph.nodes
            all_edges = codegraph.edges

        # 1. Base limits mapping (Tight, Balanced, Deep selector)
        if max_nodes <= 8:
            base_anchors, base_neighbors = 2, 4
            limit_chars = 500
        elif max_nodes <= 14:
            base_anchors, base_neighbors = 4, 8
            limit_chars = 1000
        else:
            base_anchors, base_neighbors = 8, 16
            limit_chars = 1500

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

        max_anchors = max(2, int(base_anchors * scale_factor))
        max_neighbors = max(4, int(base_neighbors * scale_factor))

        # 2. Run global PageRank centrality scoring
        pr_map = self._compute_pagerank(all_nodes, all_edges)
        
        # Calculate dynamic number of boosted PageRank hubs based on codebase size (~1% of codebase, capped between 5 and 30)
        H = max(5, min(30, int(N * 0.01)))
        
        # Sort nodes by PageRank and assign decaying calibrated boosts (scaled to exact name match weight of +8.0)
        sorted_by_pr = sorted(all_nodes, key=lambda n: pr_map.get(n.node_id, 0.0), reverse=True)
        pr_boost = {}
        for index, node in enumerate(sorted_by_pr[:H]):
            pr_boost[node.node_id] = 8.0 * (1.0 - (index / H))

        # 3. Perform dynamic Light-to-Full checks on all nodes and score with Lightweight Keyword Matcher
        terms = self._terms(query)
        
        # Traceback Line Number Scorer: extract line numbers (e.g. 'line 25' or 'file.py:25' or 'L25')
        line_numbers = []
        for match in re.finditer(r"\bline\s+(\d+)\b|:(\d+)\b|\bL(\d+)\b", query, re.IGNORECASE):
            num = match.group(1) or match.group(2) or match.group(3)
            if num:
                try:
                    line_numbers.append(int(num))
                except ValueError:
                    pass

        # Per-query file reading cache
        file_cache: dict[str, list[str]] = {}
        node_codes: dict[str, str] = {}

        node_scores = {}
        for node in all_nodes:
            code = self._read_node_code(repo_id, node, file_cache)
            node_codes[node.node_id] = code

            # Lightweight Keyword Matcher:
            keyword_score = 0.0
            for term in terms:
                # Add +10.0 if a term is found in node.label.lower()
                if node.label and term in node.label.lower():
                    keyword_score += 10.0
                # Add +5.0 if a term is found in node.file_path.lower()
                if node.file_path and term in node.file_path.lower():
                    keyword_score += 5.0

            # Add +0.2 if node.node_type is in {"function", "class", "method"}
            structural_boost = 0.0
            if node.node_type in {"function", "class", "method"}:
                structural_boost = 0.2

            # Line boundary matching: massive boost if a mentioned traceback line falls within this node's range
            line_boost = 0.0
            if node.file_path and node.line_start is not None and node.line_end is not None:
                for line_num in line_numbers:
                    if node.line_start <= line_num <= node.line_end:
                        line_boost += 15.0

            node_scores[node.node_id] = keyword_score + structural_boost + line_boost + pr_boost.get(node.node_id, 0.0)

        # 4. Select top K Primary Anchors based on final scores
        sorted_nodes = sorted(all_nodes, key=lambda n: node_scores.get(n.node_id, 0.0), reverse=True)
        relevant_nodes = [node for node in sorted_nodes if node_scores.get(node.node_id, 0.0) > 0.0]
        
        if relevant_nodes:
            anchors = relevant_nodes[:max_anchors]
        else:
            anchors = sorted_nodes[:max_anchors]
            
        selected_anchors = {node.node_id: node for node in anchors}

        # 5. Select top N Neighbor Nodes linked in the graph, ranked by score
        adjacent_edges: list[GraphEdge] = []
        neighbor_ids = set()
        edge_by_neighbor = {}
        for edge in all_edges:
            if edge.source_node in selected_anchors or edge.target_node in selected_anchors:
                adjacent_edges.append(edge)
                other_id = edge.target_node if edge.source_node in selected_anchors else edge.source_node
                if other_id not in selected_anchors:
                    neighbor_ids.add(other_id)
                    edge_by_neighbor[other_id] = edge.edge_type
                    
        node_by_id = {node.node_id: node for node in all_nodes}
        neighbor_nodes = [node_by_id[nid] for nid in neighbor_ids if nid in node_by_id]
        
        # Rank neighbors by applying connection weights to their scores
        neighbor_weighted_scores = {}
        for node in neighbor_nodes:
            edge_type = edge_by_neighbor.get(node.node_id, "imports")
            if edge_type in {"calls", "triggers", "inherits", "extends"}:
                w_edge = 1.5
            elif edge_type in {"contains", "part_of"}:
                w_edge = 1.2
            else:
                w_edge = 1.0 # imports, depends_on
                
            neighbor_weighted_scores[node.node_id] = node_scores.get(node.node_id, 0.0) * w_edge
            
        ranked_neighbors = sorted(neighbor_nodes, key=lambda n: neighbor_weighted_scores.get(n.node_id, 0.0), reverse=True)
        selected_neighbors = {node.node_id: node for node in ranked_neighbors[:max_neighbors]}

        # Combine nodes and select subset of edges
        selected_nodes = list(selected_anchors.values()) + list(selected_neighbors.values())
        selected_ids = {node.node_id for node in selected_nodes}
        selected_edges = [
            edge for edge in adjacent_edges if edge.source_node in selected_ids and edge.target_node in selected_ids
        ]

        # 6. Extract snippets: Full snippets for primary anchors, smart truncated snippets for neighbors
        snippets: list[SourceSnippet] = []
        seen_snippets = set()
        
        # Anchors (Full context)
        for node in selected_anchors.values():
            code = node_codes.get(node.node_id)
            if code is None:
                code = self._read_node_code(repo_id, node, file_cache)
            if not node.file_path and node.node_type != "cli_output":
                continue
            path = node.file_path or node.node_id
            key = (path, node.line_start, node.line_end)
            if key not in seen_snippets:
                seen_snippets.add(key)
                snippets.append(
                    SourceSnippet(
                        file_path=path,
                        line_start=node.line_start or 1,
                        line_end=node.line_end or node.line_start or 1,
                        text=code,
                        source="graph_anchor",
                    )
                )
                
        # Neighbors (Smart truncated signature context)
        for node in selected_neighbors.values():
            code = node_codes.get(node.node_id)
            if code is None:
                code = self._read_node_code(repo_id, node, file_cache)
            if not node.file_path and node.node_type != "cli_output":
                continue
            path = node.file_path or node.node_id
            key = (path, node.line_start, node.line_end)
            if key not in seen_snippets:
                seen_snippets.add(key)
                
                # Format snippet with dynamic budget limit + signature append
                truncated_text = self._format_neighbor_snippet(code, limit_chars)
                snippets.append(
                    SourceSnippet(
                        file_path=path,
                        line_start=node.line_start or 1,
                        line_end=node.line_end or node.line_start or 1,
                        text=truncated_text,
                        source="graph_neighbor",
                    )
                )

        context = self._format_context(selected_nodes, selected_edges, snippets, retrieval_method="advanced")
        measurement = self.token_service.measure_estimated("codegraph_graphify_optimized_context", context)
        
        return GraphRetrievalResult(
            context=context,
            snippets=snippets,
            selected_nodes=selected_nodes,
            selected_edges=selected_edges,
            token_measurement=measurement,
            retrieval_strategy="Advanced Hybrid Scoring System",
        )
    def _internal_retrieval(
        self, repo_id: str, query: str, max_nodes: int,
        source_selection: str, codegraph, graphify,
        graphify_mode: str = "bfs",
    ) -> GraphRetrievalResult:
        """Route to the native query engine of CodeGraph / Graphify based on source_selection.

        Only uses the external CLI tools (graphify CLI or CodeGraph Node.js).
        If the CLI fails, the exception is propagated so the user can see exactly why it failed.
        """
        selected_nodes: list[GraphNode] = []
        selected_edges: list[GraphEdge] = []

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
                graphify_cli = "graphify" # fallback

        try:
            proc = subprocess.run(
                [graphify_cli, "query", query, "--mode", graphify_mode, "--budget", str(target_budget)],
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
            "    const { CodeGraph } = require('@colbymchenry/codegraph');\n"
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
            "    console.log(JSON.stringify({ context: ctx }));\n"
            "    if (cg.close) await cg.close();\n"
            "  } catch (e) {\n"
            "    console.error(e && e.stack ? e.stack : e);\n"
            "    process.exit(2);\n"
            "  }\n"
            "})();\n"
        )
        script_path = repo_root / ".codegraph_query.js"
        script_path.write_text(script, encoding="utf-8")
        
        try:
            proc = subprocess.run(
                [node_path, str(script_path), query],
                cwd=str(repo_root),
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
            raise RuntimeError(
                f"Internal engine fails: CodeGraph CLI Node.js process failed (Exit Code {proc.returncode}). "
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
            mode_header = "### Retrieval Mode: Advanced Hybrid Engine (Lightweight Keyword + PageRank)"

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

