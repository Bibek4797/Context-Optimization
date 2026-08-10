from __future__ import annotations

import json
import re
import traceback
import streamlit as st
import time
from typing import Any, Callable, Dict, List

from app.models.schemas import QueryRecord, TokenMeasurement, CountType
from app.services.llm.base import LLMProvider
from app.services.chat_service import ChatService


class AgentHarness:
    def __init__(self, chat_service: ChatService, llm_provider: LLMProvider) -> None:
        self.chat_service = chat_service
        self.llm_provider = llm_provider
        self.tools: Dict[str, Callable[..., Any]] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        self.tools["todo_write"] = self.todo_write
        self.tools["todo_update"] = self.todo_update
        self.tools["query_codegraph"] = self.query_codegraph
        self.tools["query_langgraph"] = self.query_langgraph
        self.tools["spawn_subagent"] = self.spawn_subagent

    # ──────────────────────────────────────────────
    # Helpers: detect what assets are currently live
    # ──────────────────────────────────────────────
    @staticmethod
    def _has_codebase() -> bool:
        return bool(st.session_state.get("repo_id"))

    @staticmethod
    def _has_pdf() -> bool:
        G = st.session_state.get("unstructured_graph")
        comm = st.session_state.get("unstructured_community_summaries")
        return G is not None and bool(comm)

    # ──────────────────────────────────────────────
    # Tool Definitions
    # ──────────────────────────────────────────────
    def todo_write(self, steps: List[str]) -> str:
        """Write a step-by-step execution plan."""
        st.session_state["harness_todo"] = [
            {"step": step, "status": "pending"} for step in steps
        ]
        return "Task plan written:\n" + "\n".join(
            f"{i+1}. [pending] {step}" for i, step in enumerate(steps)
        )

    def todo_update(self, step_index: int, status: str) -> str:
        """Update the status of a plan step ('pending', 'in_progress', 'completed')."""
        if "harness_todo" not in st.session_state:
            return "Error: No active plan. Call todo_write first."
        todo_list = st.session_state["harness_todo"]
        if step_index < 0 or step_index >= len(todo_list):
            return f"Error: step_index {step_index} out of range (plan has {len(todo_list)} steps)."
        valid = ["pending", "in_progress", "completed"]
        if status not in valid:
            return f"Error: status '{status}' invalid. Must be one of {valid}."
        todo_list[step_index]["status"] = status
        st.session_state["harness_todo"] = todo_list
        return f"Step {step_index} → '{status}'."

    def query_codegraph(self, query: str) -> str:
        """Query the AST-based codebase CodeGraph for structural/dependency relationships."""
        if "retrieval_history" not in st.session_state:
            st.session_state["retrieval_history"] = []

        # Guard: must have a codebase
        if not self._has_codebase():
            err = "No codebase uploaded. Please upload a codebase ZIP in the Ingest tab first."
            st.session_state["retrieval_history"].insert(0, {
                "timestamp": time.strftime("%H:%M:%S"), "query": query,
                "type": "CodeGraph (AST)", "context_retrieved": err,
                "answer": err, "context_tokens": 0
            })
            return f"Error: {err}"

        repo_id = st.session_state["repo_id"]
        source_selection = st.session_state.get("harness_source_selection", "codegraph")
        retrieval_method = st.session_state.get("harness_retrieval_method", "internal")
        max_nodes = st.session_state.get("harness_max_nodes", 8)
        graphify_mode = st.session_state.get("harness_graphify_mode", "bfs")
        max_neighbors = st.session_state.get("harness_max_neighbors", 4)
        rectify = st.session_state.get("harness_rectify_mode", False)

        try:
            record = self.chat_service.graph_optimized_qa(
                repo_id=repo_id, query=query,
                source_selection=source_selection,
                retrieval_method=retrieval_method,
                max_nodes=max_nodes,
                graphify_mode=graphify_mode,
                max_neighbors=max_neighbors,
                rectify=rectify,
            )

            ctx_retrieved = getattr(record, "context", "") or ""
            ans_text = getattr(record, "answer", "") or ""

            nodes_info = [
                {"node_id": n.node_id, "type": n.node_type, "label": n.label,
                 "file_path": n.file_path, "line_start": n.line_start, "line_end": n.line_end}
                for n in (record.selected_nodes or [])
            ]
            edges_info = [
                {"edge_id": getattr(e, "edge_id", None) or f"{e.source_node}->{e.target_node}", "type": e.edge_type,
                 "src": e.source_node, "tgt": e.target_node}
                for e in (record.selected_edges or [])
            ]

            rec_entry = {
                "timestamp": time.strftime("%H:%M:%S"),
                "query": query,
                "type": f"CodeGraph ({source_selection.upper()})",
                "source_system": source_selection,
                "retrieval_method": retrieval_method,
                "retrieval_strategy": getattr(record, "retrieval_strategy", ""),
                "max_nodes": max_nodes,
                "max_neighbors": max_neighbors,
                "graphify_mode": graphify_mode if source_selection == "graphify" else None,
                "rectify_mode": rectify,
                "context_retrieved": ctx_retrieved,
                "selected_nodes": nodes_info,
                "selected_edges": edges_info,
                "answer": ans_text,
                "context_tokens": self.chat_service.token_service.estimate_tokens(ctx_retrieved)
            }
            st.session_state["retrieval_history"].insert(0, rec_entry)

            if record.status == "failed" and not ans_text and not ctx_retrieved:
                return f"CodeGraph query failed: {record.error}"

            return f"CodeGraph Answer for '{query}':\n{ans_text or 'Code definitions retrieved.'}\n\nContext Retrieved:\n{ctx_retrieved}"

        except Exception as e:
            err = f"Exception querying CodeGraph: {str(e)}"
            st.session_state["retrieval_history"].insert(0, {
                "timestamp": time.strftime("%H:%M:%S"), "query": query,
                "type": f"CodeGraph ({source_selection.upper()})",
                "source_system": source_selection,
                "retrieval_method": retrieval_method,
                "retrieval_strategy": "Failed Query",
                "context_retrieved": err, "answer": err, "context_tokens": 0
            })
            return err

    def query_langgraph(self, query: str) -> str:
        """Query the unstructured PDF LangGraph using Louvain community detection."""
        if "retrieval_history" not in st.session_state:
            st.session_state["retrieval_history"] = []

        # Guard: must have a built PDF graph with community summaries
        if not self._has_pdf():
            err = ("No PDF/LangGraph indexed yet. "
                   "Please upload PDFs, build LangGraph, detect communities, "
                   "and summarize them in the Ingest tab first.")
            st.session_state["retrieval_history"].insert(0, {
                "timestamp": time.strftime("%H:%M:%S"), "query": query,
                "type": "LangGraph (PDF)", "source_system": "langgraph",
                "retrieval_method": "louvain_community_detection",
                "retrieval_strategy": "Louvain Community Detection",
                "ranked_communities": [], "per_comm_details": [],
                "merged_context_prompt": err, "context_tokens": 0
            })
            return f"Error: {err}"

        G = st.session_state["unstructured_graph"]
        comm_sums = st.session_state["unstructured_community_summaries"]

        try:
            from app.services.unstructured.retrieval import run_full_query_pipeline
            embs = st.session_state.get("unstructured_community_embeddings", {})
            cache = st.session_state.get("unstructured_node_embeddings", {})

            results = run_full_query_pipeline(query, G, comm_sums, embs, cache)
            st.session_state["unstructured_node_embeddings"] = results["updated_node_embeddings_cache"]

            per_comm_details = []
            merged_context_parts = []
            for cid, score in results.get("ranked_communities", []):
                comm_ctx = results.get("per_community_contexts", {}).get(cid, {})
                per_comm_details.append({
                    "cid": cid, "score": score,
                    "summary": comm_ctx.get("summary", ""),
                    "partial_answer": comm_ctx.get("partial_answer", ""),
                    "anchors": comm_ctx.get("anchors", [])
                })
                merged_context_parts.append(
                    f"--- Community {cid} (Score: {score:.3f}) ---\n"
                    f"Summary: {comm_ctx.get('summary', '')}\n"
                    f"Intermediate Answer: {comm_ctx.get('partial_answer', '')}"
                )

            merged_text = "\n\n".join(merged_context_parts)
            st.session_state["retrieval_history"].insert(0, {
                "timestamp": time.strftime("%H:%M:%S"), "query": query,
                "type": "LangGraph (PDF)", "source_system": "langgraph",
                "retrieval_method": "louvain_community_detection",
                "retrieval_strategy": "Louvain Community Detection + Hybrid Subgraph Search",
                "ranked_communities": results.get("ranked_communities", []),
                "per_comm_details": per_comm_details,
                "merged_context_prompt": merged_text,
                "context_tokens": self.chat_service.token_service.estimate_tokens(merged_text)
            })

            return f"LangGraph Answer for '{query}':\n{results['final_answer']}"

        except Exception as e:
            err = f"Exception querying LangGraph: {str(e)}"
            st.session_state["retrieval_history"].insert(0, {
                "timestamp": time.strftime("%H:%M:%S"), "query": query,
                "type": "LangGraph (PDF)", "source_system": "langgraph",
                "retrieval_method": "louvain_community_detection",
                "retrieval_strategy": "Failed Query",
                "ranked_communities": [], "per_comm_details": [],
                "merged_context_prompt": err, "context_tokens": 0
            })
            return err

    def spawn_subagent(self, task_description: str, graph_type: str) -> str:
        """Delegate a large graph traversal to an isolated context window; returns a clean text summary."""
        if graph_type == "codegraph":
            if not self._has_codebase():
                return "Error: No codebase CodeGraph active."
            graph_doc = self.chat_service.storage.load_codegraph(st.session_state["repo_id"])
            if not graph_doc:
                return "Error: CodeGraph could not be loaded."
            nodes_text = "\n".join(
                f"- {n.node_id} ({n.node_type}) label={n.label} file={n.file_path}"
                for n in graph_doc.nodes[:100]
            )
            edges_text = "\n".join(
                f"- {e.source_node} --[{e.edge_type}]--> {e.target_node}"
                for e in graph_doc.edges[:150]
            )
            graph_text = f"Nodes:\n{nodes_text}\n\nEdges:\n{edges_text}"

        elif graph_type == "langgraph":
            if not self._has_pdf():
                return "Error: No unstructured LangGraph active."
            G = st.session_state["unstructured_graph"]
            nodes_text = "\n".join(
                f"- {node} label={G.nodes[node].get('label')} type={G.nodes[node].get('type')} "
                f"community={G.nodes[node].get('community_id')} "
                f"desc={str(G.nodes[node].get('description', ''))[:100]}"
                for node in list(G.nodes)[:100]
            )
            edges_text = "\n".join(
                f"- {u} --[{G.edges[u, v].get('relation_type')}]--> {v}"
                for u, v in list(G.edges)[:150]
            )
            graph_text = f"Nodes:\n{nodes_text}\n\nEdges:\n{edges_text}"
        else:
            return f"Error: graph_type must be 'codegraph' or 'langgraph', got '{graph_type}'."

        prompt = (
            f"You are an isolated subagent specializing in graph traversal.\n"
            f"Task: {task_description}\n\n"
            f"Graph:\n{graph_text}\n\n"
            f"Trace relationships and return ONLY a clean text summary of your findings."
        )
        try:
            response = self.llm_provider.generate_answer(prompt)
            return f"Subagent Summary:\n{response.text}"
        except Exception as e:
            return f"Exception in subagent: {str(e)}"

    # ──────────────────────────────────────────────
    # Master Execution Loop
    # ──────────────────────────────────────────────
    def execute(
        self,
        user_query: str,
        max_iterations: int = 6,
        source_preference: str = "auto",
        callback: Any = None,
        chat_history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Perception-Action-Observation loop.

        Scenario routing (determined at call time, before any LLM call):
          • Only codebase  → only query_codegraph is available
          • Only PDF       → only query_langgraph is available
          • Both           → both tools available, preference guided by source_preference
          • Neither        → direct LLM answer, no graph tools needed
        """
        has_code = self._has_codebase()
        has_pdf  = self._has_pdf()

        # Build Multi-Turn Chat History context string if provided
        chat_history_str = ""
        if chat_history:
            turns = []
            for msg in chat_history[-6:]:  # Keep 6 recent turns
                role = msg.get("role", "user").capitalize()
                content = str(msg.get("content", ""))
                # Strip out long XML code blocks for concise history
                content_clean = re.sub(r"<code_fix>.*?</code_fix>", "[Code Fix Applied]", content, flags=re.DOTALL)
                turns.append(f"{role}: {content_clean[:400]}")
            if turns:
                chat_history_str = "### Previous Conversation Context:\n" + "\n".join(turns) + "\n\n"

        # ── Scenario classification ──────────────────
        if has_code and has_pdf:
            scenario = "both"
        elif has_code:
            scenario = "code_only"
        elif has_pdf:
            scenario = "pdf_only"
        else:
            scenario = "none"

        # ── Initialise harness checklist ─────────────
        if scenario == "both":
            initial_steps = [
                f"Understand query: '{user_query[:50]}'",
                "Query CodeGraph for structural/code context",
                "Query LangGraph for PDF/document context",
                "Synthesize merged final answer",
            ]
        elif scenario == "code_only":
            initial_steps = [
                f"Understand query: '{user_query[:50]}'",
                "Query CodeGraph for code structure and dependencies",
                "Synthesize final answer from code context",
            ]
        elif scenario == "pdf_only":
            initial_steps = [
                f"Understand query: '{user_query[:50]}'",
                "Query LangGraph communities for document facts",
                "Synthesize final answer from document context",
            ]
        else:
            initial_steps = [
                f"Understand query: '{user_query[:50]}'",
                "Answer directly from LLM knowledge (no graphs uploaded)",
            ]

        st.session_state["harness_todo"] = [
            {"step": s, "status": "pending"} for s in initial_steps
        ]
        st.session_state["harness_todo"][0]["status"] = "in_progress"

        if callback:
            callback([], st.session_state["harness_todo"])

        # ── Fast-path: no graphs at all ──────────────
        if scenario == "none":
            try:
                direct_prompt = (
                    f"Answer the following question using your own knowledge.\n"
                    f"{chat_history_str}"
                    f"Question: {user_query}"
                )
                resp = self.llm_provider.generate_answer(direct_prompt)
                final_answer = resp.text.strip()
            except Exception as e:
                final_answer = f"Error generating direct answer: {str(e)}"

            for item in st.session_state["harness_todo"]:
                item["status"] = "completed"
            if callback:
                callback([], st.session_state["harness_todo"])
            return {"query": user_query, "final_answer": final_answer, "iterations": 0, "history": []}

        # ── Build scenario-specific system prompt ────
        computed_pref = source_preference
        if source_preference == "auto":
            code_keywords = {
                "code", "function", "method", "class", "def", "import", "package", 
                "bug", "fix", "error", "exception", "test", "implementation", 
                "call", "variable", "parameter", "return", "syntax", "line", 
                "file", "script", "program", "develop", "repository", "repo",
                "git", "refactor", "patch", "modify", "correct", "rectify",
                "loop", "statement", "dependency", "module", "inherits", "extends"
            }
            doc_keywords = {
                "rule", "guideline", "policy", "document", "pdf", "manual", 
                "instruction", "standard", "requirement", "specification", 
                "compliance", "doc", "text", "description", "summary", 
                "overview", "concept", "theory", "explain", "meaning"
            }
            query_words = set(re.findall(r"\w+", user_query.lower()))
            code_score = len(query_words.intersection(code_keywords))
            doc_score = len(query_words.intersection(doc_keywords))
            
            if code_score > doc_score:
                computed_pref = "code_first"
            elif doc_score > code_score:
                computed_pref = "pdf_first"
            else:
                computed_pref = "auto"

        rectify = st.session_state.get("harness_rectify_mode", False)
        system_prompt = self._build_system_prompt(scenario, computed_pref, rectify=rectify)

        # ── PAO Loop ─────────────────────────────────
        history: List[Dict] = []
        final_answer: str | None = None
        consecutive_errors = 0
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # ── 92% Context Window Compression ───────
            history_lines_check = [
                f"Thought:{s.get('thought','')} Action:{s.get('tool','')} Obs:{s.get('observation','')}"
                for s in history
            ]
            current_size = len("\n".join(history_lines_check))

            if current_size >= 11040 and len(history) > 3:
                first_step = history[0] if history[0].get("tool") == "todo_write" else None
                middle = history[1:-2] if first_step else history[:-2]
                recent = history[-2:]
                compressed_obs = "Compressed prior iterations:\n" + "\n".join(
                    f"- {m.get('tool')} → {str(m.get('observation',''))[:180]}..."
                    for m in middle
                )
                compressed = {
                    "thought": "History compressed at 92% context threshold.",
                    "tool": "history_compressor",
                    "tool_input": "{}",
                    "observation": compressed_obs
                }
                history = ([first_step] if first_step else []) + [compressed] + recent

            # ── Build prompt ──────────────────────────
            history_text = "\n".join(
                f"### Iteration {i+1}\n"
                f"Thought: {s.get('thought','')}\n"
                f"Action: {s.get('tool','')} inputs={s.get('tool_input','{}')}\n"
                f"Observation: {s.get('observation','')}\n"
                for i, s in enumerate(history)
            )

            # Safeguard: if a retrieval tool has already returned valid context, instruct LLM to produce final_answer immediately
            has_retrieval_obs = any(
                step.get("tool") in ("query_codegraph", "query_langgraph") 
                and "Error:" not in step.get("observation", "")
                for step in history
            )
            force_answer_note = ""
            if has_retrieval_obs:
                force_answer_note = "\n\nCRITICAL: Graph retrieval results are already available in your Execution History above. Do NOT call retrieval tools again. Produce your JSON response with 'final_answer' NOW."

            prompt = (
                f"{system_prompt}\n\n"
                f"{chat_history_str}"
                f"User Question: {user_query}\n\n"
                f"### Execution History:\n{history_text}"
                f"{force_answer_note}\n\n"
                f"### Next Step:\nOutput only a valid JSON object."
            )

            # ── Call LLM ─────────────────────────────
            try:
                llm_resp = self.llm_provider.generate_answer(prompt)
                response_text = llm_resp.text.strip()
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                obs = f"LLM call failed: {str(e)}"
                history.append({"thought": "LLM error.", "tool": "none", "tool_input": "{}", "observation": obs})
                if callback:
                    callback(history, st.session_state.get("harness_todo", []))
                if consecutive_errors >= 3:
                    final_answer = (
                        f"Harness aborted after 3 consecutive LLM failures. "
                        f"Check your API key / rate limits. Last error: {str(e)}"
                    )
                    break
                time.sleep(3.0)
                continue

            # ── Parse JSON ────────────────────────────
            parsed: dict = {}
            try:
                clean = re.sub(r"^```(?:json)?\n?", "", response_text.strip())
                clean = re.sub(r"\n?```$", "", clean).strip()
                parsed = json.loads(clean)
            except Exception:
                # Try to extract first JSON object
                s = response_text.find("{"); e = response_text.rfind("}")
                if s != -1 and e != -1:
                    try:
                        parsed = json.loads(response_text[s:e+1])
                    except Exception:
                        pass
            if not parsed:
                obs = f"Non-JSON output received: {response_text[:300]}"
                history.append({"thought": "Bad LLM output.", "tool": "none", "tool_input": "{}", "observation": obs})
                if callback:
                    callback(history, st.session_state.get("harness_todo", []))
                continue

            thought = parsed.get("thought", "")

            # ── Final answer? ─────────────────────────
            if "final_answer" in parsed:
                final_answer = parsed["final_answer"]
                for item in st.session_state.get("harness_todo", []):
                    item["status"] = "completed"
                history.append({"thought": thought, "tool": "none", "tool_input": "{}", "observation": "Final answer produced."})
                if callback:
                    callback(history, st.session_state.get("harness_todo", []))
                break

            # ── Tool call ─────────────────────────────
            tool_name = parsed.get("tool")
            tool_input = parsed.get("tool_input", {})

            if not tool_name:
                obs = "JSON must contain 'final_answer' or 'tool'."
                history.append({"thought": thought, "tool": "none", "tool_input": "{}", "observation": obs})
                if callback:
                    callback(history, st.session_state.get("harness_todo", []))
                continue

            # Guard: prevent the LLM calling a tool that doesn't apply to the current scenario
            if tool_name == "query_codegraph" and not has_code:
                obs = "Tool 'query_codegraph' is unavailable: no codebase has been uploaded."
                history.append({"thought": thought, "tool": tool_name, "tool_input": json.dumps(tool_input), "observation": obs})
                if callback:
                    callback(history, st.session_state.get("harness_todo", []))
                continue

            if tool_name == "query_langgraph" and not has_pdf:
                obs = "Tool 'query_langgraph' is unavailable: no PDF LangGraph has been built."
                history.append({"thought": thought, "tool": tool_name, "tool_input": json.dumps(tool_input), "observation": obs})
                if callback:
                    callback(history, st.session_state.get("harness_todo", []))
                continue

            if tool_name not in self.tools:
                obs = f"Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}."
            else:
                try:
                    obs = self.tools[tool_name](**tool_input) if isinstance(tool_input, dict) else self.tools[tool_name](tool_input)
                    # Auto-advance the first non-completed checklist step
                    if tool_name not in ("todo_write", "todo_update"):
                        todo = st.session_state.get("harness_todo", [])
                        for item in todo:
                            if item.get("status") in ("pending", "in_progress"):
                                item["status"] = "completed"
                                st.session_state["harness_todo"] = todo
                                break
                except Exception as exc:
                    obs = f"Error executing '{tool_name}': {str(exc)}\n{traceback.format_exc()}"

            history.append({
                "thought": thought,
                "tool": tool_name,
                "tool_input": json.dumps(tool_input),
                "observation": str(obs)
            })
            if callback:
                callback(history, st.session_state.get("harness_todo", []))
            time.sleep(0.4)

        if final_answer is None:
            final_answer = (
                f"Harness reached the {max_iterations}-iteration limit without a final answer. "
                f"Try narrowing your question or uploading the required assets."
            )

        # Safeguard: if Rectify Mode is enabled and a tool returned a code fix in history,
        # but the planner failed to include it in final_answer, we auto-append it!
        if st.session_state.get("harness_rectify_mode", False) and final_answer:
            tool_fixes = []
            for step in history:
                obs = step.get("observation", "")
                if "<code_fix>" in obs:
                    normalized = obs.replace("\\\\n", "\n").replace("\\n", "\n")
                    pattern = re.compile(
                        r"<code_fix>[\s\n\r]*"
                        r"<filepath>(?P<filepath>.*?)</filepath>[\s\n\r]*"
                        r"<original_code>(?P<original>.*?)</original_code>[\s\n\r]*"
                        r"<replacement_code>(?P<replacement>.*?)</replacement_code>[\s\n\r]*"
                        r"</code_fix>",
                        re.IGNORECASE | re.DOTALL,
                    )
                    for m in pattern.finditer(normalized):
                        tool_fixes.append({
                            "filepath": m.group("filepath").strip(),
                            "original": m.group("original").strip(),
                            "replacement": m.group("replacement").strip(),
                        })
            
            if tool_fixes and "<code_fix>" not in final_answer.lower():
                append_blocks = []
                for fix in tool_fixes:
                    append_blocks.append(
                        f"\n\n<code_fix>\n"
                        f"  <filepath>{fix['filepath']}</filepath>\n"
                        f"  <original_code>\n{fix['original']}\n  </original_code>\n"
                        f"  <replacement_code>\n{fix['replacement']}\n  </replacement_code>\n"
                        f"</code_fix>"
                    )
                final_answer += "".join(append_blocks)

        return {
            "query": user_query,
            "final_answer": final_answer,
            "iterations": iteration,
            "history": history
        }

    # ──────────────────────────────────────────────
    # Dynamic System Prompt (scenario-aware)
    # ──────────────────────────────────────────────
    def _build_system_prompt(self, scenario: str, source_preference: str, rectify: bool = False) -> str:
        """
        Build the LLM system prompt based on exactly what assets are available.
        This prevents the LLM from hallucinating tool calls for missing assets.
        """

        # Tool descriptions shown conditionally
        codegraph_desc = (
            "1. `query_codegraph`: Query the AST-based codebase graph for structural "
            "relationships, function calls, class hierarchies, and file dependencies.\n"
            "   Args: {\"query\": \"your query string\"}"
        )
        langgraph_desc = (
            "2. `query_langgraph`: Query the PDF document graph using Louvain community "
            "detection to retrieve relevant document facts and guidelines.\n"
            "   Args: {\"query\": \"your query string\"}"
        )
        spawn_desc = (
            "3. `spawn_subagent`: Delegate a complex graph traversal to an isolated subagent "
            "that returns a clean summary. Only use for very large traversals.\n"
            "   Args: {\"task_description\": \"...\", \"graph_type\": \"codegraph\" | \"langgraph\"}"
        )
        plan_desc = (
            "0. `todo_write`: Write a step-by-step execution plan BEFORE querying. "
            "For simple single-topic queries you MAY skip this.\n"
            "   Args: {\"steps\": [\"step 1\", \"step 2\", ...]}"
        )

        if scenario == "code_only":
            tools_section = (
                "## Available Tools\n"
                f"{plan_desc}\n"
                f"{codegraph_desc}\n"
                f"{spawn_desc}\n\n"
                "IMPORTANT: Only `query_codegraph` and `spawn_subagent` (graph_type='codegraph') "
                "are available. The user has uploaded a codebase only. "
                "Do NOT call `query_langgraph` — no PDF has been indexed."
            )
            strategy = "Answer from the codebase CodeGraph. Focus on structural dependencies, function relationships, and file-level analysis."

        elif scenario == "pdf_only":
            tools_section = (
                "## Available Tools\n"
                f"{plan_desc}\n"
                f"{langgraph_desc}\n"
                f"{spawn_desc}\n\n"
                "IMPORTANT: Only `query_langgraph` and `spawn_subagent` (graph_type='langgraph') "
                "are available. The user has uploaded PDFs only. "
                "Do NOT call `query_codegraph` — no codebase has been indexed."
            )
            strategy = "Answer from the PDF LangGraph. Focus on document facts, guidelines, and community-detected themes."

        else:  # both
            if source_preference == "code_first":
                priority = "Prefer `query_codegraph` first. Then supplement with `query_langgraph` if additional document context is needed."
            elif source_preference == "pdf_first":
                priority = "Prefer `query_langgraph` first. Then supplement with `query_codegraph` if additional code context is needed."
            else:  # auto
                priority = (
                    "Choose the tool that best matches the query intent: "
                    "use `query_codegraph` for code/structural questions and "
                    "`query_langgraph` for document/guideline questions. "
                    "For hybrid questions, query both."
                )
            tools_section = (
                "## Available Tools\n"
                f"{plan_desc}\n"
                f"{codegraph_desc}\n"
                f"{langgraph_desc}\n"
                f"{spawn_desc}\n\n"
                f"Source Selection Strategy: {priority}"
            )
            strategy = "Merge insights from both the codebase CodeGraph and the PDF LangGraph to provide a comprehensive answer."

        rectify_instructions = ""
        if rectify:
            rectify_instructions = (
                "\n\n## Code Fix Preservation (Rectify Mode)\n"
                "If any tool (e.g. `query_codegraph`) returns a `<code_fix>` XML block in its observation, "
                "you MUST copy and include that exact `<code_fix>...</code_fix>` XML structure "
                "verbatim in your `final_answer`. Do NOT modify, strip, or alter the XML tags."
            )

        return f"""You are the central Agentic Harness Planner for a Context Optimization Engine.
Your goal is to answer the user's query by querying the correct graph knowledge base(s).

{tools_section}

## Answer Strategy
{strategy}{rectify_instructions}

## Output Format
Every response must be a single valid JSON object — one of:

Tool call:
{{
  "thought": "Why you are calling this tool.",
  "tool": "tool_name",
  "tool_input": {{"arg": "value"}}
}}

Final answer (when you have enough information):
{{
  "thought": "Final synthesis reasoning.",
  "final_answer": "Your direct, clear answer to the user. No meta-commentary, no jargon like 'based on community 3' or 'the AST shows'. Answer as a senior engineer."
}}

## Critical Rules
- Do NOT output anything outside the JSON object.
- Do NOT call a tool more than twice for the same query.
- Do NOT hallucinate graph data — only report what the tool observations actually return.
- If a tool returns an error, report it in final_answer; do not retry indefinitely.
- Keep final_answer concise and professional.
"""
