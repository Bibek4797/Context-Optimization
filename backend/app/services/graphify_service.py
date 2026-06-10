from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from app.models.schemas import GraphDocument, GraphEdge, GraphNode


class GraphifyService:
    def __init__(self, storage: Any, *args: Any, **kwargs: Any) -> None:
        self.storage = storage


    def run_or_fallback(self, repo_id: str, repo_root: Path, codegraph: GraphDocument) -> GraphDocument:
        # Build Native Graphify directly from CodeGraph in Python.
        # This completely eliminates the external graphify CLI subprocess dependency.
        
        # 1. Map child nodes to parent nodes based on 'contains' edges in CodeGraph
        parent_map: dict[str, str] = {}
        for edge in codegraph.edges:
            if edge.edge_type == "contains":
                parent_map[edge.target_node] = edge.source_node

        # 2. Build a type map of CodeGraph nodes to look up their original categories
        node_type_map: dict[str, str] = {node.node_id: node.node_type for node in codegraph.nodes}

        # Helper: Find the closest macro-level ancestor (module/class/etc.) for any node
        def get_macro_ancestor(node_id: str) -> str | None:
            current = node_id
            visited = set()
            while current:
                if current in visited:
                    break
                visited.add(current)
                ntype = node_type_map.get(current)
                # Macro nodes are files, classes, components, imports, and external symbols (not functions/methods)
                if ntype and ntype not in {"function", "method"}:
                    return current
                current = parent_map.get(current)
            return None

        # 3. Filter out micro-nodes (functions/methods) and only map macro nodes to Graphify
        nodes: list[GraphNode] = []
        node_map: dict[str, str] = {} # maps CodeGraph macro node_id to Graphify node_id
        
        for node in codegraph.nodes:
            # Skip micro-level logic nodes (functions and methods) to achieve macro-level pruning
            if node.node_type in {"function", "method"}:
                continue

            graphify_id = f"graphify:{node.node_id.replace('codegraph:', '')}"
            node_map[node.node_id] = graphify_id
            
            # Map structural AST node types to design-level concepts
            node_type = "concept"
            if node.node_type == "module":
                node_type = "file"
            elif node.node_type in {"class", "struct_item"}:
                node_type = "component"
                
            nodes.append(
                GraphNode(
                    node_id=graphify_id,
                    node_type=node_type,
                    label=node.label,
                    file_path=node.file_path,
                    line_start=node.line_start,
                    line_end=node.line_end,
                    source_snippet=None,
                    metadata={**node.metadata, "graphify_processed": True, "native": True},
                )
            )

        # 4. Convert and lift CodeGraph structural relationships to high-level flow edges.
        # Micro-level dependencies (like functions calling other functions) are aggregated/lifted to their macro ancestors.
        unique_edges: dict[tuple[str, str, str], GraphEdge] = {}
        
        for edge in codegraph.edges:
            # Find the nearest macro ancestor of both the source and target nodes
            macro_src = get_macro_ancestor(edge.source_node)
            macro_tgt = get_macro_ancestor(edge.target_node)
            
            if not macro_src or not macro_tgt:
                continue
            
            # Skip self-loop dependencies at the macro-level (e.g. methods calling within the same class/file)
            if macro_src == macro_tgt:
                continue
                
            source_id = f"graphify:{macro_src.replace('codegraph:', '')}"
            target_id = f"graphify:{macro_tgt.replace('codegraph:', '')}"
            
            # Map structural AST edge types to dynamic execution flows
            edge_type = "flow"
            if edge.edge_type == "contains":
                edge_type = "part_of"
            elif edge.edge_type == "imports":
                edge_type = "depends_on"
            elif edge.edge_type == "calls":
                edge_type = "triggers"
            elif edge.edge_type == "inherits":
                edge_type = "extends"
                
            edge_key = (source_id, target_id, edge_type)
            graphify_edge_id = f"graphify:edge:{macro_src.replace('codegraph:', '')}:{macro_tgt.replace('codegraph:', '')}:{edge_type}"
            
            # Keep the edge with the highest score if duplicates arise during lifting
            if edge_key not in unique_edges or unique_edges[edge_key].score < edge.score:
                unique_edges[edge_key] = GraphEdge(
                    edge_id=graphify_edge_id,
                    edge_type=edge_type,
                    source_node=source_id,
                    target_node=target_id,
                    score=edge.score,
                    metadata={**edge.metadata, "graphify_processed": True, "native": True, "lifted": True},
                )
            
        return GraphDocument(
            repo_id=repo_id,
            source="graphify",
            nodes=nodes,
            edges=list(unique_edges.values()),
            raw_output_path=None,
            warnings=[],
        )
