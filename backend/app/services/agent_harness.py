from __future__ import annotations

import json
import traceback
import streamlit as st
import time
from typing import Any, Callable, Dict, List
import networkx as nx

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
        # Register tools with string names matching user instructions
        self.tools["todo_write"] = self.todo_write
        self.tools["todo_update"] = self.todo_update
        self.tools["query_codegraph"] = self.query_codegraph
        self.tools["query_langgraph"] = self.query_langgraph
        self.tools["spawn_subagent"] = self.spawn_subagent

    # --- Tool Definitions ---

    def todo_write(self, steps: List[str]) -> str:
        """Use this tool to write a step-by-step plan before executing cross-graph queries."""
        st.session_state["harness_todo"] = [
            {"step": step, "status": "pending"} for step in steps
        ]
        return "Task plan written successfully. Current plan:\n" + "\n".join(
            f"{i}. [pending] {step}" for i, step in enumerate(steps)
        )

    def todo_update(self, step_index: int, status: str) -> str:
        """Use this tool to update the status of a step (e.g. 'completed', 'in_progress', 'pending')."""
        if "harness_todo" not in st.session_state:
            return "Error: No active plan found. Write a plan first using todo_write."
        todo_list = st.session_state["harness_todo"]
        if step_index < 0 or step_index >= len(todo_list):
            return f"Error: Invalid step index {step_index}. Plan has {len(todo_list)} steps."
        
        valid_statuses = ["pending", "in_progress", "completed"]
        if status not in valid_statuses:
            return f"Error: Invalid status '{status}'. Must be one of {valid_statuses}."
            
        todo_list[step_index]["status"] = status
        st.session_state["harness_todo"] = todo_list
        return f"Step {step_index} updated to status '{status}' successfully."

    def query_codegraph(self, query: str) -> str:
        """Use this tool to map functional relationships and dependencies from codebase source files."""
        repo_id = st.session_state.get("repo_id")
        if not repo_id:
            return "Error: No repository uploaded/active. User must upload a repository first."
        try:
            # We call the underlying CodeGraph QA pipeline
            record = self.chat_service.graph_optimized_qa(
                repo_id=repo_id,
                query=query,
                source_selection="codegraph",
                retrieval_method="internal"
            )
            if record.status == "failed":
                return f"Error querying CodeGraph: {record.error}"
                
            # Log CodeGraph retrieval details
            if "retrieval_history" not in st.session_state:
                st.session_state["retrieval_history"] = []
            
            st.session_state["retrieval_history"].insert(0, {
                "timestamp": time.strftime("%H:%M:%S"),
                "query": query,
                "type": "CodeGraph (AST)",
                "context_retrieved": record.context or "No explicit source context retrieved.",
                "answer": record.answer
            })
            
            return f"CodeGraph Answer for '{query}':\n{record.answer}\n\nContext Retrieved:\n{record.context}"
        except Exception as e:
            return f"Exception querying CodeGraph: {str(e)}"

    def query_langgraph(self, query: str) -> str:
        """Use this tool strictly for retrieving unstructured knowledge from PDFs using community detection."""
        # Check if the unstructured graph exists in session state
        G = st.session_state.get("unstructured_graph")
        comm_sums = st.session_state.get("unstructured_community_summaries")
        
        if G is None or not comm_sums:
            return "Error: No PDF/unstructured document indexed yet. User must upload and build the PDF graph first."
            
        try:
            from app.services.unstructured.retrieval import run_full_query_pipeline
            embs = st.session_state.get("unstructured_community_embeddings", {})
            cache = st.session_state.get("unstructured_node_embeddings", {})
            
            results = run_full_query_pipeline(query, G, comm_sums, embs, cache)
            
            # Save updated cache back
            st.session_state["unstructured_node_embeddings"] = results["updated_node_embeddings_cache"]
            
            # Log LangGraph retrieval details
            if "retrieval_history" not in st.session_state:
                st.session_state["retrieval_history"] = []
                
            per_comm_details = []
            merged_context_parts = []
            for cid, score in results.get("ranked_communities", []):
                comm_ctx = results.get("per_community_contexts", {}).get(cid, {})
                summary_text = comm_ctx.get("summary", "")
                partial_ans = comm_ctx.get("partial_answer", "")
                anchors = comm_ctx.get("anchors", [])
                
                per_comm_details.append({
                    "cid": cid,
                    "score": score,
                    "summary": summary_text,
                    "partial_answer": partial_ans,
                    "anchors": anchors
                })
                
                merged_context_parts.append(
                    f"--- Community {cid} (Relevance Score: {score:.3f}) ---\n"
                    f"Summary: {summary_text}\n"
                    f"Intermediate Answer: {partial_ans}"
                )
                
            st.session_state["retrieval_history"].insert(0, {
                "timestamp": time.strftime("%H:%M:%S"),
                "query": query,
                "type": "LangGraph (PDF)",
                "ranked_communities": results.get("ranked_communities", []),
                "per_comm_details": per_comm_details,
                "merged_context_prompt": "\n\n".join(merged_context_parts)
            })
            
            return f"LangGraph Answer for '{query}':\n{results['final_answer']}"
        except Exception as e:
            return f"Exception querying LangGraph: {str(e)}"

    def spawn_subagent(self, task_description: str, graph_type: str) -> str:
        """Use this tool to delegate massive graph traversals to an isolated context window, returning only a clean text summary to the main loop."""
        # Identify the target graph
        if graph_type == "codegraph":
            repo_id = st.session_state.get("repo_id")
            if not repo_id:
                return "Error: No codebase CodeGraph active."
            graph_doc = self.chat_service.storage.load_codegraph(repo_id)
            if not graph_doc:
                return "Error: CodeGraph could not be loaded."
            
            # Serialize CodeGraph nodes and edges for traversal
            nodes_text = "\n".join(
                f"- Node: {n.node_id} ({n.node_type}), Label={n.label}, File={n.file_path}"
                for n in graph_doc.nodes[:100]  # Cap to prevent token limit issues
            )
            edges_text = "\n".join(
                f"- {e.source_node} --[{e.edge_type}]--> {e.target_node}"
                for e in graph_doc.edges[:150]
            )
            graph_text = f"Nodes:\n{nodes_text}\n\nEdges:\n{edges_text}"
            
        elif graph_type == "langgraph":
            G = st.session_state.get("unstructured_graph")
            if G is None:
                return "Error: No unstructured LangGraph active."
            
            # Serialize unstructured graph
            nodes_text = "\n".join(
                f"- Node: {node}, Label={G.nodes[node].get('label')}, Type={G.nodes[node].get('type')}, Community={G.nodes[node].get('community_id')}, Desc={G.nodes[node].get('description')[:120]}..."
                for node in list(G.nodes)[:100]
            )
            edges_text = "\n".join(
                f"- {u} --[{G.edges[u, v].get('relation_type')}]--> {v}"
                for u, v in list(G.edges)[:150]
            )
            graph_text = f"Nodes:\n{nodes_text}\n\nEdges:\n{edges_text}"
        else:
            return f"Error: Invalid graph_type '{graph_type}'. Must be 'codegraph' or 'langgraph'."

        prompt = f"""You are an isolated subagent specializing in graph traversal and analysis.
Your task is: {task_description}

Here is the serialized graph representation:
{graph_text}

Perform the traversal, trace relationships, and summarize the findings. Return ONLY a clean text summary of your traversal path and findings. Do not output JSON or conversational fluff.
"""
        try:
            response = self.llm_provider.generate_answer(prompt)
            return f"Subagent Traversal Summary for task '{task_description}':\n{response.text}"
        except Exception as e:
            return f"Exception during subagent execution: {str(e)}"

    # --- Master Loop ---

    def execute(self, user_query: str, max_iterations: int = 16, callback: Callable[[List[Dict[str, str]], List[Dict[str, Any]]], None] | None = None) -> Dict[str, Any]:
        """Runs the perception-action-observation master loop."""
        history: List[Dict[str, str]] = []
        iteration = 0
        final_answer = None
        consecutive_errors = 0
        
        # Reset plan in state
        st.session_state["harness_todo"] = []
        
        while iteration < max_iterations:
            iteration += 1
            
            # 1. Construct prompt showing history
            history_lines = []
            for step_idx, step in enumerate(history):
                history_lines.append(f"### Iteration {step_idx + 1}")
                history_lines.append(f"Thought: {step.get('thought', '')}")
                history_lines.append(f"Action: Call tool '{step.get('tool', '')}' with inputs {step.get('tool_input', '')}")
                history_lines.append(f"Observation: {step.get('observation', '')}")
                history_lines.append("")
                
            history_text = "\n".join(history_lines)
            
            system_prompt = self._get_system_prompt()
            
            prompt = f"""{system_prompt}

User Question: {user_query}

### Loop Execution History:
{history_text}

### Next Step:
Provide the JSON response for the current iteration. Remember, if you have not written a plan yet, you must use `todo_write`.
"""
            
            # 2. Call LLM
            try:
                llm_response = self.llm_provider.generate_answer(prompt)
                response_text = llm_response.text.strip()
                consecutive_errors = 0  # Reset on success
            except Exception as e:
                consecutive_errors += 1
                observation = f"Error calling LLM provider: {str(e)}"
                history.append({
                    "thought": "LLM generation failed, attempting to retry.",
                    "tool": "none",
                    "tool_input": "{}",
                    "observation": observation
                })
                if callback:
                    callback(history, st.session_state.get("harness_todo", []))
                
                if consecutive_errors >= 3:
                    final_answer = f"The Agent Harness was aborted after 3 consecutive LLM API failures. Please check your API key, billing status, or rate limits. Last error: {str(e)}"
                    break
                    
                time.sleep(3.0)  # Cool down to let rate limits clear
                continue

            # 3. Parse JSON from LLM
            parsed_response = {}
            try:
                # Strip markdown code blocks if present
                clean_text = response_text
                if clean_text.startswith("```"):
                    clean_text = re.sub(r"^```(?:json)?\n", "", clean_text)
                    clean_text = re.sub(r"\n```$", "", clean_text)
                clean_text = clean_text.strip()
                parsed_response = json.loads(clean_text)
            except Exception as e:
                # Find JSON block manually
                start = response_text.find("{")
                end = response_text.rfind("}")
                if start != -1 and end != -1:
                    try:
                        parsed_response = json.loads(response_text[start:end+1])
                    except Exception:
                        pass
                if not parsed_response:
                    observation = f"Error: Output was not valid JSON. Response received: {response_text}. Please return only valid JSON."
                    history.append({
                        "thought": "LLM returned malformed output.",
                        "tool": "none",
                        "tool_input": "{}",
                        "observation": observation
                    })
                    if callback:
                        callback(history, st.session_state.get("harness_todo", []))
                    continue

            # 4. Process response
            thought = parsed_response.get("thought", "")
            
            # Check if final answer is returned
            if "final_answer" in parsed_response:
                final_answer = parsed_response["final_answer"]
                history.append({
                    "thought": thought,
                    "tool": "none",
                    "tool_input": "{}",
                    "observation": "Final answer generated."
                })
                if callback:
                    callback(history, st.session_state.get("harness_todo", []))
                break
                
            # Process tool call
            tool_name = parsed_response.get("tool")
            tool_input = parsed_response.get("tool_input", {})
            
            if not tool_name:
                observation = "Error: JSON must contain either 'final_answer' or a 'tool' to call."
                history.append({
                    "thought": thought,
                    "tool": "none",
                    "tool_input": "{}",
                    "observation": observation
                })
                if callback:
                    callback(history, st.session_state.get("harness_todo", []))
                continue
                
            # Dispatch tool
            observation = ""
            if tool_name not in self.tools:
                observation = f"Error: Tool '{tool_name}' is not registered. Available tools: {list(self.tools.keys())}."
            else:
                # Catch all errors in try-except block so the loop never breaks
                try:
                    # Execute tool function
                    if isinstance(tool_input, dict):
                        observation = self.tools[tool_name](**tool_input)
                    else:
                        observation = self.tools[tool_name](tool_input)
                except Exception as exc:
                    observation = f"Error executing tool '{tool_name}': {str(exc)}\n{traceback.format_exc()}"
            
            # Log step
            history.append({
                "thought": thought,
                "tool": tool_name,
                "tool_input": json.dumps(tool_input),
                "observation": str(observation)
            })
            
            if callback:
                callback(history, st.session_state.get("harness_todo", []))
                
            # Sleep a bit and force rerun in UI so it updates live
            time.sleep(0.5)
            # st.rerun() can be triggered by the caller to refresh the page/sidebar
            
        if final_answer is None:
            final_answer = f"The Agent Harness exceeded the limit of {max_iterations} iterations without finding a final answer. Please narrow down your query."
            
        return {
            "query": user_query,
            "final_answer": final_answer,
            "iterations": iteration,
            "history": history
        }

    def _get_system_prompt(self) -> str:
        return """You are the central Agentic Harness Planner. Your goal is to answer the user's query by coordinating information from a codebase CodeGraph and an unstructured document LangGraph.

You have access to a Tool Dispatch Registry with the following tools:
1. `todo_write`: Write a step-by-step execution plan.
   Args: {"steps": ["step 1", "step 2", ...]}
2. `todo_update`: Update a plan step status.
   Args: {"step_index": int, "status": "pending" | "in_progress" | "completed"}
3. `query_codegraph`: Query the codebase CodeGraph for structure/dependency relationships.
   Args: {"query": "string query"}
4. `query_langgraph`: Query the unstructured document LangGraph for PDF text facts using community detection.
   Args: {"query": "string query"}
5. `spawn_subagent`: Delegate a massive graph traversal to an isolated context window, returning a clean text summary.
   Args: {"task_description": "string describing the traversal", "graph_type": "codegraph" | "langgraph"}

CRITICAL INSTRUCTIONS:
- You MUST write a step-by-step plan using `todo_write` BEFORE executing any other queries.
- As you complete each step in your plan, update its status to "completed" using `todo_update`. (Updating to "in_progress" is optional if you want to save loop iterations).
- Do not make assumptions. Query the appropriate graph using the tools.
- All actions must be output as valid JSON matching the format below.

Output Format:
Your response must be a single, valid JSON object containing either a tool call or the final answer.
To call a tool, output:
{
  "thought": "Your thought process about why you are calling this tool.",
  "tool": "tool_name",
  "tool_input": {
    "arg1": "value1"
  }
}

To return the final answer, output:
{
  "thought": "Your final synthesis thought.",
  "final_answer": "Your detailed answer to the user's question, citing the CodeGraph and LangGraph sources where appropriate."
}

Do not include any text before or after the JSON object. Do not wrap it in markdown code blocks unless they are standard json blocks (i.e. starting with ```json).
"""

import re
