from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from app.models.schemas import GraphDocument, GraphEdge, GraphNode, RepoFile
from app.services.file_utils import read_text_lossy, source_snippet

EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".ipynb": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}

CLASS_NODE_TYPES = {"class_definition", "class_declaration", "class", "type_spec", "struct_item"}
FUNCTION_NODE_TYPES = {
    "function_definition", "function_declaration", "arrow_function", 
    "method_definition", "function", "method_declaration", 
    "function_item"
}
IMPORT_NODE_TYPES = {"import_statement", "import_from_statement", "import_spec", "import_declaration", "use_declaration"}
CALL_NODE_TYPES = {"call", "call_expression", "method_invocation"}


class CodeGraphService:

    def build(self, repo_id: str, repo_root: Path, files: list[RepoFile]) -> GraphDocument:
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}
        symbol_index: dict[str, str] = {}
        module_index: dict[str, str] = {}
        pending_imports: list[tuple[str, str, str]] = []
        pending_calls: list[tuple[str, str, int | None]] = []
        pending_inherits: list[tuple[str, str]] = []
        warnings: list[str] = []

        for repo_file in files:
            rel_path = repo_file.path
            path = repo_root / rel_path
            text = read_text_lossy(path)
            source_bytes = text.encode("utf-8")
            
            module_name = self._module_name(rel_path)
            module_id = self._node_id("module", rel_path, module_name, 1)
            
            # Module source snippet capped at 800 characters
            module_snippet = text[:800] + "\n..." if len(text) > 800 else text
            module_node = GraphNode(
                node_id=module_id,
                node_type="module",
                label=module_name,
                file_path=rel_path,
                line_start=1,
                line_end=max(1, repo_file.line_count),
                source_snippet=module_snippet,
                metadata={"path": rel_path},
            )
            nodes[module_id] = module_node
            symbol_index[module_name] = module_id
            module_index[module_name] = module_id

            # Load dynamic parser for the file language
            parser_res = self._get_parser_for_file(rel_path)
            if not parser_res or not parser_res[0]:
                warnings.append(f"Tree-sitter parser not available for {rel_path}; skipping detailed AST build.")
                continue

            parser, lang_name = parser_res
            try:
                tree = parser.parse(source_bytes)
                root_node = tree.root_node
            except Exception as exc:
                warnings.append(f"Tree-sitter parse failed for {rel_path}: {exc}")
                continue

            parent_stack: list[tuple[str, str]] = [("module", module_id)]
            
            # Walk top level children to extract CodeGraph relationships
            for child in root_node.children:
                try:
                    self._traverse_tree(
                        child,
                        source_bytes,
                        rel_path,
                        module_name,
                        parent_stack,
                        nodes,
                        edges,
                        symbol_index,
                        pending_imports,
                        pending_calls,
                        pending_inherits,
                    )
                except Exception as exc:
                    warnings.append(f"Error traversing AST node in {rel_path}: {exc}")

        # Resolve imports
        for source_id, import_name, imported_module in pending_imports:
            target_id = module_index.get(imported_module) or module_index.get(import_name)
            if target_id is None:
                target_id = self._node_id("import", imported_module, import_name, 0)
                nodes.setdefault(
                    target_id,
                    GraphNode(
                        node_id=target_id,
                        node_type="import",
                        label=import_name,
                        metadata={"module": imported_module, "external": True},
                    ),
                )
            self._add_edge(edges, "imports", source_id, target_id, score=1.0)

        # Resolve calls
        GENERIC_BUILTINS = {
            "print", "len", "range", "dict", "list", "set", "tuple", "int", "str", "float", "bool",
            "enumerate", "zip", "sum", "min", "max", "abs", "round", "super", "getattr", "setattr",
            "hasattr", "isinstance", "issubclass", "append", "split", "join", "replace", "strip",
            "format", "open", "read", "write", "close", "any", "all", "map", "filter", "next",
            "iter", "sorted", "reversed", "dir", "type", "id", "hash", "repr", "callable", "write_bytes",
            "read_text", "write_text", "get", "post", "put", "delete", "request"
        }
        for source_id, call_name, line_no in pending_calls:
            short = call_name.split(".")[-1]
            if short.lower() in GENERIC_BUILTINS or call_name.lower() in GENERIC_BUILTINS:
                continue
                
            target_id = symbol_index.get(call_name)
            if target_id is None:
                target_id = symbol_index.get(short)
            if target_id is None:
                target_id = self._node_id("external_symbol", call_name, call_name, line_no or 0)
                nodes.setdefault(
                    target_id,
                    GraphNode(
                        node_id=target_id,
                        node_type="external_symbol",
                        label=call_name,
                        metadata={"external": True},
                    ),
                )
            self._add_edge(edges, "calls", source_id, target_id, score=0.7)

        # Resolve inherits
        for class_id, base_name in pending_inherits:
            target_id = symbol_index.get(base_name) or symbol_index.get(base_name.split(".")[-1])
            if target_id is None:
                target_id = self._node_id("external_symbol", base_name, base_name, 0)
                nodes.setdefault(
                    target_id,
                    GraphNode(
                        node_id=target_id,
                        node_type="external_symbol",
                        label=base_name,
                        metadata={"external": True},
                    ),
                )
            self._add_edge(edges, "inherits", class_id, target_id, score=0.8)

        return GraphDocument(
            repo_id=repo_id,
            source="codegraph",
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            warnings=warnings,
        )

    def _get_parser_for_file(self, file_path: str):
        suffix = Path(file_path).suffix.lower()
        lang_name = EXTENSION_TO_LANGUAGE.get(suffix, "python")
        
        from tree_sitter import Language, Parser
        
        try:
            if lang_name == "python":
                import tree_sitter_python as lang_module
                lang_func = lang_module.language
            elif lang_name == "javascript":
                import tree_sitter_javascript as lang_module
                lang_func = lang_module.language
            elif lang_name == "typescript":
                import tree_sitter_typescript as lang_module
                if hasattr(lang_module, "language_typescript"):
                    lang_func = lang_module.language_typescript
                elif hasattr(lang_module, "language"):
                    lang_func = lang_module.language
                else:
                    lang_func = lang_module.language_tsx
            elif lang_name == "go":
                import tree_sitter_go as lang_module
                lang_func = lang_module.language
            elif lang_name == "rust":
                import tree_sitter_rust as lang_module
                lang_func = lang_module.language
            elif lang_name == "java":
                import tree_sitter_java as lang_module
                lang_func = lang_module.language
            elif lang_name == "cpp":
                import tree_sitter_cpp as lang_module
                lang_func = lang_module.language
            elif lang_name == "c":
                import tree_sitter_c as lang_module
                lang_func = lang_module.language
            else:
                import tree_sitter_python as lang_module
                lang_func = lang_module.language
            
            language = Language(lang_func())
            parser = Parser()
            try:
                parser.language = language
            except AttributeError:
                parser.set_language(language)
            return parser, lang_name
        except (ImportError, AttributeError, Exception):
            try:
                import tree_sitter_python as lang_module
                language = Language(lang_module.language())
                parser = Parser()
                try:
                    parser.language = language
                except AttributeError:
                    parser.set_language(language)
                return parser, "python"
            except Exception:
                return None, None


    def _traverse_tree(
        self,
        node,
        source_bytes: bytes,
        rel_path: str,
        module_name: str,
        parent_stack: list[tuple[str, str]],
        nodes: dict[str, GraphNode],
        edges: dict[str, GraphEdge],
        symbol_index: dict[str, str],
        pending_imports: list[tuple[str, str, str]],
        pending_calls: list[tuple[str, str, int | None]],
        pending_inherits: list[tuple[str, str]],
    ) -> None:
        parent_kind, parent_id = parent_stack[-1]
        node_type = node.type
        
        if node_type in IMPORT_NODE_TYPES:
            text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
            words = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", text)
            if len(words) > 1:
                for word in words[1:]:
                    if word not in {"import", "from", "as", "require", "export"}:
                        pending_imports.append((parent_id, word, words[0]))
            return
            
        elif node_type in CLASS_NODE_TYPES:
            name = None
            for child in node.children:
                if child.type == "identifier":
                    name = source_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
                    break
            if not name:
                name = f"Class_L{node.start_point[0] + 1}"
                
            qualname = self._qualname(module_name, parent_stack, nodes, name)
            graph_node = self._create_symbol_node("class", rel_path, source_bytes, name, qualname, node)
            nodes[graph_node.node_id] = graph_node
            symbol_index[name] = graph_node.node_id
            symbol_index[qualname] = graph_node.node_id
            self._add_edge(edges, "contains", parent_id, graph_node.node_id, score=1.0)
            
            for child in node.children:
                if child.type in {"argument_list", "superclasses", "extends_interfaces", "type_list"}:
                    bases = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", source_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace"))
                    for base in bases:
                        pending_inherits.append((graph_node.node_id, base))
                        
            parent_stack.append(("class", graph_node.node_id))
            for child in node.children:
                try:
                    self._traverse_tree(
                        child, source_bytes, rel_path, module_name, parent_stack,
                        nodes, edges, symbol_index, pending_imports, pending_calls, pending_inherits
                    )
                except Exception:
                    pass
            parent_stack.pop()
            return
            
        elif node_type in FUNCTION_NODE_TYPES:
            name = None
            for child in node.children:
                if child.type in {"identifier", "field_identifier"}:
                    name = source_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
                    break
            if not name:
                name = f"Func_L{node.start_point[0] + 1}"
                
            qualname = self._qualname(module_name, parent_stack, nodes, name)
            kind = "method" if parent_kind == "class" else "function"
            graph_node = self._create_symbol_node(kind, rel_path, source_bytes, name, qualname, node)
            nodes[graph_node.node_id] = graph_node
            symbol_index[name] = graph_node.node_id
            symbol_index[qualname] = graph_node.node_id
            self._add_edge(edges, "contains", parent_id, graph_node.node_id, score=1.0)
            
            try:
                self._find_calls_in_node(node, source_bytes, graph_node.node_id, pending_calls)
            except Exception:
                pass
            return

        for child in node.children:
            try:
                self._traverse_tree(
                    child, source_bytes, rel_path, module_name, parent_stack,
                    nodes, edges, symbol_index, pending_imports, pending_calls, pending_inherits
                )
            except Exception:
                pass

    def _find_calls_in_node(self, node, source_bytes: bytes, caller_id: str, pending_calls: list) -> None:
        if node.type in CALL_NODE_TYPES:
            call_name = None
            for child in node.children:
                if child.type in {"identifier", "attribute", "field_expression", "member_expression"}:
                    call_name = source_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
                    break
            if call_name:
                pending_calls.append((caller_id, call_name, node.start_point[0] + 1))
                
        for child in node.children:
            try:
                self._find_calls_in_node(child, source_bytes, caller_id, pending_calls)
            except Exception:
                pass

    def _create_symbol_node(
        self,
        node_type: str,
        rel_path: str,
        source_bytes: bytes,
        label: str,
        qualname: str,
        node,
    ) -> GraphNode:
        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1
        
        segment = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        if len(segment) > 800:
            segment = segment[:800] + "\n..."
            
        return GraphNode(
            node_id=self._node_id(node_type, rel_path, qualname, line_start),
            node_type=node_type,
            label=label,
            file_path=rel_path,
            line_start=line_start,
            line_end=line_end,
            source_snippet=segment,
            metadata={"qualified_name": qualname, "tree_sitter_type": node.type},
        )

    def _module_name(self, rel_path: str) -> str:
        path = Path(rel_path)
        parts = list(path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts) or path.stem

    def _qualname(
        self,
        module_name: str,
        parent_stack: list[tuple[str, str]],
        nodes: dict[str, GraphNode],
        name: str,
    ) -> str:
        class_parts = [nodes[node_id].label for kind, node_id in parent_stack if kind == "class" and node_id in nodes]
        return ".".join([module_name, *class_parts, name])

    def _node_id(self, node_type: str, rel_path: str, label: str, line: int) -> str:
        raw = f"{node_type}:{rel_path}:{label}:{line}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        return f"codegraph:{node_type}:{digest}:{label}"

    def _edge_id(self, edge_type: str, source_id: str, target_id: str) -> str:
        raw = f"{edge_type}:{source_id}:{target_id}"
        return f"codegraph:edge:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"

    def _add_edge(
        self,
        edges: dict[str, GraphEdge],
        edge_type: str,
        source_id: str,
        target_id: str,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        edge_id = self._edge_id(edge_type, source_id, target_id)
        edges.setdefault(
            edge_id,
            GraphEdge(
                edge_id=edge_id,
                edge_type=edge_type,
                source_node=source_id,
                target_node=target_id,
                score=score,
                metadata=metadata or {},
            ),
        )
