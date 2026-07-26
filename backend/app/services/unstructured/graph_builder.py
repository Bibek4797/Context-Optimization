import json
import re
import networkx as nx
import streamlit as st
from app.services.unstructured import llm_client

def parse_json_from_llm(response_text: str) -> dict:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    cleaned = cleaned.strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
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
            
        # Process nodes
        for node_data in doc_nodes:
            node_id = str(node_data.get("id")).strip().lower().replace(" ", "_")
            if not node_id:
                continue
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
        
        # Process edges
        for edge_data in doc_edges:
            src = str(edge_data.get("source_id")).strip().lower().replace(" ", "_")
            tgt = str(edge_data.get("target_id")).strip().lower().replace(" ", "_")
            rel = edge_data.get("relation_type", "associated_with")
            
            if not src or not tgt or src == tgt:
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
