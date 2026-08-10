import networkx as nx
from app.services.unstructured import llm_client

def detect_communities(G: nx.Graph, resolution: float = 1.0) -> dict:
    if not G or len(G) == 0:
        return {}
    
    G_undirected = G.to_undirected()
    
    # ── Hierarchical Multi-Level Louvain Community Trees ──
    # Level 0 (Global Themes): gamma = 0.5
    # Level 1 (Mid Topics):    gamma = 1.0 (Default)
    # Level 2 (Micro Facts):   gamma = 2.5
    
    def _run_louvain(res_val: float) -> dict:
        comm_map = {}
        try:
            from networkx.algorithms import community as nx_comm
            comm_sets = nx_comm.louvain_communities(G_undirected, resolution=res_val, seed=42)
            for cid, node_set in enumerate(comm_sets):
                for node in node_set:
                    comm_map[node] = cid
        except Exception:
            for node in G.nodes:
                comm_map[node] = 0
        return comm_map

    level_0_map = _run_louvain(0.5)
    level_1_map = _run_louvain(resolution)
    level_2_map = _run_louvain(2.5)

    for node in G.nodes:
        G.nodes[node]["community_level_0"] = level_0_map.get(node, 0)
        G.nodes[node]["community_id"] = level_1_map.get(node, 0) # Level 1 (Default)
        G.nodes[node]["community_level_2"] = level_2_map.get(node, 0)
        
    return level_1_map

def group_nodes_by_community(G: nx.Graph, level_key: str = "community_id") -> dict[int, list]:
    communities = {}
    for node, data in G.nodes(data=True):
        cid = data.get(level_key, 0)
        if cid not in communities:
            communities[cid] = []
        communities[cid].append(node)
    return communities

def summarize_community(G: nx.Graph, community_id: int, node_ids: list, max_nodes: int = 50) -> str:
    if not llm_client.is_configured():
        return "API key not configured."
        
    node_details = []
    for nid in node_ids[:max_nodes]:
        nd = G.nodes[nid]
        label = nd.get("label", nid)
        ntype = nd.get("type", "General")
        desc = nd.get("description", "")
        node_details.append(f"Entity: {label} (Type: {ntype}) - Description: {desc}")
        
    edge_details = []
    sub_g = G.subgraph(node_ids)
    for u, v, data in list(sub_g.edges(data=True))[:max_nodes]:
        u_label = G.nodes[u].get("label", u)
        v_label = G.nodes[v].get("label", v)
        rel = data.get("relation_type", "associated_with")
        edge_details.append(f"{u_label} --[{rel}]--> {v_label}")
        
    nodes_str = "\n".join(node_details)
    edges_str = "\n".join(edge_details)
    
    prompt = f"""You are a research analyst summarizing a semantic community from a knowledge graph.
    
Below is the list of entities and relationships inside community {community_id}:

Entities:
{nodes_str}

Relationships:
{edges_str}

Provide a concise summary (3 to 6 sentences) of this community. Highlight the central themes, key entities, and how they relate.
Do not include markdown headers or list formatting in your response. Just write the paragraphs.
"""
    try:
        summary = llm_client.generate_text(prompt)
    except Exception as e:
        print(f"[Community Summarization Error] LLM generation failed: {e}")
        summary = ""
        
    if not summary:
        summary = f"This community contains entities: {', '.join([G.nodes[n].get('label', n) for n in node_ids[:10]])}."
    return summary

def summarize_all_communities(G: nx.Graph) -> tuple[dict[int, str], dict[int, list[float]]]:
    # Check if nodes already have community_id. If not, detect them.
    has_communities = any("community_id" in data for _, data in G.nodes(data=True))
    if not has_communities:
        detect_communities(G)
        
    # Summarize Level 1 communities (Mid Topics)
    communities = group_nodes_by_community(G, level_key="community_id")
    
    summaries = {}
    embeddings = {}
    
    for cid, node_ids in communities.items():
        summary = summarize_community(G, cid, node_ids)
        summaries[cid] = summary
        embeddings[cid] = []
            
    return summaries, embeddings
