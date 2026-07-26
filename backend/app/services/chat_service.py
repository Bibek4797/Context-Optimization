from __future__ import annotations

import time
from uuid import uuid4

from typing import Any

from app.models.schemas import CountType, QueryRecord, TokenMeasurement
from app.services.graph_retrieval_service import GraphRetrievalService
from app.services.llm.base import LLMConfigurationError, LLMProvider
from app.services.storage import LocalStorage
from app.services.token_service import TokenService


class ChatService:
    def __init__(
        self,
        storage: LocalStorage,
        graph_retrieval_service: GraphRetrievalService,
        token_service: TokenService,
        llm_provider: LLMProvider,
        pipeline: Any = None,
    ) -> None:
        self.storage = storage
        self.graph_retrieval_service = graph_retrieval_service
        self.token_service = token_service
        self.llm_provider = llm_provider
        
        # Instantiate RectificationService natively
        from app.services.rectification_service import RectificationService
        self.rectification_service = RectificationService(storage, pipeline)

    def apply_rectification(self, repo_id: str, file_path: str, original_code: str, replacement_code: str) -> dict[str, Any]:
        return self.rectification_service.apply_code_fix(repo_id, file_path, original_code, replacement_code)



    def graph_optimized_qa(
        self, repo_id: str, query: str, session_id: str | None = None,
        source_selection: str = "codegraph", max_nodes: int = 8,
        rectify: bool = False, retrieval_method: str = "internal",
        graphify_mode: str = "bfs", max_anchors: int | None = None,
        max_neighbors: int | None = None,
    ) -> QueryRecord:
        if self.storage.load_repo_metadata(repo_id) is None:
            raise ValueError("Repo not found.")
        session_id = session_id or uuid4().hex
        query_id = uuid4().hex
        started = time.perf_counter()
        
        answer = ""
        error = None
        status = "completed"
        token_usage = {}
        snippets = []
        selected_nodes = []
        selected_edges = []
        retrieval_strategy = "unknown"
        context = ""
        
        try:
            graph_context = self.graph_retrieval_service.build_context(
                repo_id, 
                query, 
                max_nodes=max_nodes, 
                source_selection=source_selection,
                retrieval_method=retrieval_method,
                graphify_mode=graphify_mode,
                max_anchors=max_anchors,
                max_neighbors=max_neighbors
            )
            snippets = graph_context.snippets
            selected_nodes = graph_context.selected_nodes
            selected_edges = graph_context.selected_edges
            retrieval_strategy = graph_context.retrieval_strategy
            context = graph_context.context
            token_usage["codegraph_graphify_optimized_context"] = graph_context.token_measurement
            
            prompt = self._graph_prompt(query, graph_context.context, rectify)
            token_usage["llm_prompt_tokens"] = self._prompt_measurement(prompt)
            
            llm_response = self.llm_provider.generate_answer(prompt)
            answer = llm_response.text
            token_usage["llm_prompt_tokens"] = llm_response.prompt_tokens
            token_usage["llm_response_tokens"] = llm_response.response_tokens
            token_usage["total_per_query_tokens"] = llm_response.total_tokens
        except Exception as exc:
            status = "failed"
            error = str(exc)
            context = f"Error building context: {exc}"
            
            # Populate dummy token measurements so save/render doesn't fail
            if "codegraph_graphify_optimized_context" not in token_usage:
                token_usage["codegraph_graphify_optimized_context"] = TokenMeasurement(
                    stage="codegraph_graphify_optimized_context",
                    tokens=0,
                    count_type=CountType.exact,
                    notes=f"Error building context: {exc}"
                )
            if "llm_prompt_tokens" not in token_usage:
                token_usage["llm_prompt_tokens"] = TokenMeasurement(
                    stage="llm_prompt_tokens",
                    tokens=0,
                    count_type=CountType.exact,
                    notes="Failed before prompt construction."
                )
            token_usage["llm_response_tokens"] = TokenMeasurement(
                stage="llm_response_tokens",
                tokens=0,
                count_type=CountType.exact,
                provider="gemini",
                notes="No response generated due to retrieval failure.",
            )
            token_usage["total_per_query_tokens"] = TokenMeasurement(
                stage="total_per_query_tokens",
                tokens=0,
                count_type=CountType.exact,
                provider="gemini",
                notes="Query execution failed.",
            )
            
        # Compute Whole Codebase Baseline (General Chatbot approach: Concatenated whole repository code files)
        try:
            repo_tokens = self._get_total_repo_tokens(repo_id)
            query_prompt = self._standard_prompt(query, "[concatenated_files_placeholder]")
            query_tokens = self._prompt_measurement(query_prompt).tokens
            baseline_total = repo_tokens + query_tokens
        except Exception:
            baseline_total = 10000 # Safety fallback
            
        token_usage["whole_codebase_baseline"] = TokenMeasurement(
            stage="whole_codebase_baseline",
            tokens=baseline_total,
            count_type=CountType.exact if self.token_service._encoding else CountType.estimated,
            notes="Prompt tokens required if sending 100% of codebase files directly to LLM."
        )

        # Normalize token_usage entries to dicts to prevent Pydantic V2 model type validation mismatches
        normalized_token_usage = {}
        for k, v in token_usage.items():
            if hasattr(v, "model_dump"):
                normalized_token_usage[k] = v.model_dump()
            elif isinstance(v, dict):
                normalized_token_usage[k] = v
            else:
                normalized_token_usage[k] = v

        record = QueryRecord(
            query_id=query_id,
            repo_id=repo_id,
            session_id=session_id,
            mode="graph_optimized",
            query=query,
            status=status,
            answer=answer,
            error=error,
            source_snippets=snippets,
            selected_nodes=selected_nodes,
            selected_edges=selected_edges,
            token_usage=normalized_token_usage,
            retrieval_strategy=retrieval_strategy,
            context=context,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        self.storage.save_query(record)
        self.storage.append_log(repo_id, "chat-graph", "info", f"Query {query_id} finished with status {status}.")
        return record



    def _prompt_measurement(self, prompt: str) -> TokenMeasurement:
        try:
            return self.llm_provider.count_tokens(prompt, "llm_prompt_tokens")
        except LLMConfigurationError as exc:
            return self.token_service.measure_estimated("llm_prompt_tokens", prompt, notes=str(exc))



    def _graph_prompt(self, query: str, context: str, rectify: bool = False) -> str:
        rectify_str = ""
        if rectify:
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
        return (
            "You are an expert repository software architect. Use the selected CodeGraph and Graphify context to answer. "
            "CRITICAL FORMATTING: Answer the question directly, clearly, and concisely. "
            "Do NOT include meta-commentary, self-referential filler, or technical jargon such as 'based on the graph context', 'derived from nodes', or 'according to the AST traversal'. "
            "Provide a clean, direct, natural response."
            f"{rectify_str}\n\n"
            f"Question:\n{query}\n\n"
            f"Optimized graph context:\n{context}"
        )

    def _get_total_repo_tokens(self, repo_id: str) -> int:
        meta = self.storage.load_repo_metadata(repo_id)
        if meta and meta.stats:
            if hasattr(meta.stats, "total_tokens") and meta.stats.total_tokens > 0:
                return meta.stats.total_tokens

        from app.services.file_utils import read_text_lossy
        repo_root = self.storage.repo_source_dir(repo_id)
        if not repo_root.exists():
            return 0
        
        files = self.storage.load_files(repo_id)
        total_tokens = 0
        for repo_file in files:
            file_path = repo_root / repo_file.path
            if file_path.exists():
                try:
                    text = read_text_lossy(file_path)
                    total_tokens += self.token_service.estimate_tokens(text)
                except Exception:
                    continue
        return total_tokens


