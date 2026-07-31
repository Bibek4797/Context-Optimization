import json
import re
import networkx as nx
import streamlit as st
from app.services.unstructured import llm_client

def parse_json_from_llm(response_text: str) -> dict | list:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    cleaned = cleaned.strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract a JSON object {...}
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start:end+1])
            except json.JSONDecodeError:
                pass
        # Try to extract a JSON array [...]
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start:end+1])
            except json.JSONDecodeError:
                pass
        raise ValueError("Could not decode JSON from response.")


def extract_graph_from_text(text: str) -> dict:
    if not llm_client.is_configured():
        raise ValueError("LLM API key is not configured. Please configure it in the sidebar.")
        
    prompt = f"""You are a knowledge graph extractor. Analyze the following text and exhaustively extract ALL important entities (nodes) and ALL significant relationships (edges) between them. Do not limit the extraction; be thorough, detailed, and comprehensive so that the resulting knowledge graph captures the full semantic content of the text.
        
Text:
{text}

You MUST return a JSON object with exactly the following structure:
{{
  "nodes": [
    {{
      "id": "unique_string_id_no_spaces",
      "label": "Entity Name",
      "type": "Entity Type (e.g. Person, Organization, Concept, Event, Variable, System, Component)",
      "description": "Detailed explanation of the entity in this context"
    }}
  ],
  "edges": [
    {{
      "source_id": "unique_string_id_no_spaces_of_source",
      "target_id": "unique_string_id_no_spaces_of_target",
      "relation_type": "The relationship (e.g. employee_of, part_of, created_by, acts_on, influences)"
    }}
  ]
}}

Ensure all node IDs are consistent (e.g., if "Google" is referred to in multiple places, use the same id like "google").
Return ONLY valid JSON. Do not include any other conversational text.
"""
    try:
        response = llm_client.generate_text(prompt, json_mode=True)
        
        # If response is empty (e.g. rate limit error occurred)
        if not response or not response.strip():
            raise ValueError("Empty response received from LLM API.")
            
        if "raw_extractions" not in st.session_state:
            st.session_state["raw_extractions"] = []
        st.session_state["raw_extractions"].append(response)
        
        return parse_json_from_llm(response)
    except Exception as e:
        raise RuntimeError(f"API failure during entity extraction: {e}")

try:
    from thefuzz import fuzz
except ImportError:
    fuzz = None

def find_fuzzy_canonical_node(G: nx.Graph, node_id: str, threshold: int = 90) -> str:
    """Finds an existing node in G that fuzzy-matches node_id to merge duplicates (e.g. 'andrew_ng' vs 'prof_andrew_ng')."""
    if G.has_node(node_id):
        return node_id
    if fuzz is None or len(node_id) < 4:
        return node_id
    node_str = node_id.replace("_", " ")
    for existing_node in list(G.nodes):
        if len(existing_node) >= 4:
            existing_str = existing_node.replace("_", " ")
            score = fuzz.token_set_ratio(node_str, existing_str)
            if score >= threshold:
                return existing_node
    return node_id

def build_graph_from_documents(documents: list[dict]) -> nx.Graph:
    G = nx.Graph()
    
    for doc in documents:
        text = doc["text"]
        if not text.strip():
            continue
            
        # Dynamically scale chunk size and overlap depending on document length (target ~7 chunks, capped between 1,000 and 15,000)
        file_len = len(text)
        chunk_size = min(15000, max(1000, file_len // 7))
        overlap = chunk_size // 10
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            if end < len(text):
                # Search backward for natural word boundary (space or sentence end) to avoid slicing words in half
                space_idx = text.rfind(" ", start + (chunk_size // 2), end)
                if space_idx != -1:
                    end = space_idx
                    
            chunks.append(text[start:end])
            
            if end >= len(text):
                break
                
            next_start = end - overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start
                
        total_chunks = len(chunks)
        
        doc_nodes = []
        doc_edges = []
        
        for idx, chunk in enumerate(chunks):
            if total_chunks > 1:
                st.info(f"Processing chunk {idx+1}/{total_chunks} of document '{doc['name']}'...")
            
            data = extract_graph_from_text(chunk)
            doc_nodes.extend(data.get("nodes", []))
            doc_edges.extend(data.get("edges", []))
            
        # Process nodes (with fuzzy deduplication)
        for node_data in doc_nodes:
            raw_id = str(node_data.get("id")).strip().lower().replace(" ", "_")
            if not raw_id:
                continue
            
            # Resolve canonical ID via fuzzy matching if an equivalent node exists
            node_id = find_fuzzy_canonical_node(G, raw_id, threshold=90)
            
            label = node_data.get("label", node_id)
            ntype = node_data.get("type", "General")
            desc = node_data.get("description", "")
            
            if G.has_node(node_id):
                existing_desc = G.nodes[node_id].get("description", "")
                if desc and desc not in existing_desc:
                    G.nodes[node_id]["description"] = (existing_desc + "; " + desc).strip("; ")
                G.nodes[node_id]["doc_ids"].add(doc["id"])
            else:
                G.add_node(
                    node_id,
                    label=label,
                    type=ntype,
                    description=desc,
                    doc_ids={doc["id"]}
                )
        
        # Process edges (resolving canonical fuzzy node IDs for source and target)
        for edge_data in doc_edges:
            raw_src = str(edge_data.get("source_id")).strip().lower().replace(" ", "_")
            raw_tgt = str(edge_data.get("target_id")).strip().lower().replace(" ", "_")
            rel = edge_data.get("relation_type", "associated_with")
            
            if not raw_src or not raw_tgt or raw_src == raw_tgt:
                continue
                
            src = find_fuzzy_canonical_node(G, raw_src, threshold=90)
            tgt = find_fuzzy_canonical_node(G, raw_tgt, threshold=90)
            
            if src == tgt:
                continue
            
            if not G.has_node(src):
                G.add_node(src, label=src.replace("_", " ").title(), type="General", description="", doc_ids={doc["id"]})
            if not G.has_node(tgt):
                G.add_node(tgt, label=tgt.replace("_", " ").title(), type="General", description="", doc_ids={doc["id"]})
                
            if G.has_edge(src, tgt):
                existing_rel = G.edges[src, tgt].get("relation_type", "")
                if rel not in existing_rel:
                    G.edges[src, tgt]["relation_type"] = (existing_rel + ", " + rel).strip(", ")
                G.edges[src, tgt]["doc_ids"].add(doc["id"])
            else:
                G.add_edge(
                    src,
                    tgt,
                    relation_type=rel,
                    doc_ids={doc["id"]}
                )
                
    # Convert sets to lists for json safety
    for node in G.nodes:
        G.nodes[node]["doc_ids"] = list(G.nodes[node]["doc_ids"])
    for u, v in G.edges:
        G.edges[u, v]["doc_ids"] = list(G.edges[u, v]["doc_ids"])
        
    return G

def sample_graph(G: nx.Graph, max_nodes: int = 100) -> nx.Graph:
    if len(G) <= max_nodes:
        return G
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, deg in degrees[:max_nodes]]
    return G.subgraph(top_nodes)
