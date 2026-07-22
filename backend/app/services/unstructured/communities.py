import networkx as nx
from app.services.unstructured import llm_client

def detect_communities(G: nx.Graph, resolution: float = 1.0) -> dict:
    if not G or len(G) == 0:
        return {}
    
    G_undirected = G.to_undirected()
    community_map = {}
    
    try:
        from networkx.algorithms import community as nx_comm
        res = resolution
        comm_sets = nx_comm.louvain_communities(G_undirected, resolution=res, seed=42)
        
        # If Louvain groups nodes into too few communities, dynamically scale resolution to force partitioning
        while len(comm_sets) < 3 and res < 15.0 and len(G_undirected.nodes) >= 10:
            res += 1.0
            comm_sets = nx_comm.louvain_communities(G_undirected, resolution=res, seed=42)
            
        for cid, node_set in enumerate(comm_sets):
            for node in node_set:
                community_map[node] = cid
    except Exception:
        try:
            from networkx.algorithms import community as nx_comm
            comm_sets = list(nx_comm.greedy_modularity_communities(G_undirected))
            for cid, node_set in enumerate(comm_sets):
                for node in node_set:
                    community_map[node] = cid
        except Exception:
            try:
                comp_sets = list(nx.connected_components(G_undirected))
                for cid, node_set in enumerate(comp_sets):
                    for node in node_set:
                        community_map[node] = cid
            except Exception:
                for node in G.nodes:
                    community_map[node] = 0
                    
    for node in G.nodes:
        G.nodes[node]["community_id"] = community_map.get(node, 0)
        
    return community_map

def group_nodes_by_community(G: nx.Graph) -> dict[int, list]:
    communities = {}
    for node, data in G.nodes(data=True):
        cid = data.get("community_id", 0)
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
        
    communities = group_nodes_by_community(G)
    
    summaries = {}
    embeddings = {}
    
    for cid, node_ids in communities.items():
        summary = summarize_community(G, cid, node_ids)
        summaries[cid] = summary
        
        # Skip API-based embedding generation to conserve quota; matching is done locally via TF-IDF & SVD (LSA)
        embeddings[cid] = []
            
    return summaries, embeddings
