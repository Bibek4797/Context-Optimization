import numpy as np
import networkx as nx
import re
import streamlit as st
from app.services.unstructured import llm_client

def cosine_similarity(v1, v2):
    if not v1 or not v2:
        return 0.0
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot = np.dot(v1, v2)
    norm_a = np.linalg.norm(v1)
    norm_b = np.linalg.norm(v2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def select_relevant_communities_tfidf_lsa(question: str, community_summaries: dict[int, str], top_k: int = 3) -> list[tuple[int, float]]:
    import math
    import collections
    
    def tokenize(text):
        return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
        
    cids = list(community_summaries.keys())
    documents = [tokenize(community_summaries[cid]) for cid in cids]
    query = tokenize(question)
    
    if not documents or not query:
        return [(cid, 0.0) for cid in cids][:top_k]
        
    # Build Vocabulary
    vocab = sorted(list(set(word for doc in documents for word in doc)))
    if not vocab:
        return [(cid, 0.0) for cid in cids][:top_k]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Calculate IDF
    N = len(documents)
    idf = {}
    for word in vocab:
        df = sum(1 for doc in documents if word in doc)
        idf[word] = math.log(1.0 + (N / (1.0 + df)))
        
    # Represent documents as TF-IDF vectors
    doc_vectors = np.zeros((N, len(vocab)))
    for doc_idx, doc in enumerate(documents):
        counts = collections.Counter(doc)
        doc_len = len(doc) if len(doc) > 0 else 1
        for word, count in counts.items():
            if word in word_to_idx:
                tf = count / doc_len
                doc_vectors[doc_idx, word_to_idx[word]] = tf * idf[word]
                
    # Represent query as TF-IDF vector
    query_vector = np.zeros(len(vocab))
    query_counts = collections.Counter(query)
    query_len = len(query) if len(query) > 0 else 1
    for word, count in query_counts.items():
        if word in word_to_idx:
            tf = count / query_len
            query_vector[word_to_idx[word]] = tf * idf[word]
            
    # Apply SVD (LSA) if we have enough dimensions/documents
    k_components = 5
    if N > 2 and len(vocab) > k_components:
        try:
            # SVD: doc_vectors is (N, vocab_size)
            # Keep top k_components
            U, S, Vt = np.linalg.svd(doc_vectors, full_matrices=False)
            k = min(N, len(vocab), k_components)
            U_k = U[:, :k]
            S_k = np.diag(S[:k])
            Vt_k = Vt[:k, :]
            
            # Project document and query vectors into latent space
            doc_vectors_projected = U_k @ S_k
            query_vector_projected = query_vector @ Vt_k.T
            
            scores = []
            q_norm = np.linalg.norm(query_vector_projected)
            for doc_idx, cid in enumerate(cids):
                d_vec = doc_vectors_projected[doc_idx]
                d_norm = np.linalg.norm(d_vec)
                if q_norm > 0 and d_norm > 0:
                    sim = float(np.dot(query_vector_projected, d_vec) / (q_norm * d_norm))
                else:
                    sim = 0.0
                scores.append((cid, sim))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]
        except Exception:
            pass
            
    # Standard TF-IDF Cosine Similarity Fallback
    scores = []
    q_norm = np.linalg.norm(query_vector)
    for doc_idx, cid in enumerate(cids):
        d_vec = doc_vectors[doc_idx]
        d_norm = np.linalg.norm(d_vec)
        if q_norm > 0 and d_norm > 0:
            sim = float(np.dot(query_vector, d_vec) / (q_norm * d_norm))
        else:
            sim = 0.0
        scores.append((cid, sim))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def select_relevant_communities_cooc_svd(question: str, community_summaries: dict[int, str], top_k: int = 3) -> list[tuple[int, float]]:
    import math
    import collections
    
    def tokenize(text):
        return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
        
    cids = list(community_summaries.keys())
    documents = [tokenize(community_summaries[cid]) for cid in cids]
    query = tokenize(question)
    
    if not documents or not query:
        return [(cid, 0.0) for cid in cids][:top_k]
        
    # Build Vocabulary
    vocab = sorted(list(set(word for doc in documents for word in doc)))
    if not vocab:
        return [(cid, 0.0) for cid in cids][:top_k]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    V = len(vocab)
    
    # 1. Build Word Co-occurrence Matrix (V x V) with a sliding window of 2 words
    cooc_matrix = np.zeros((V, V))
    window_size = 2
    for doc in documents:
        for i, word in enumerate(doc):
            if word not in word_to_idx:
                continue
            w_idx = word_to_idx[word]
            start = max(0, i - window_size)
            end = min(len(doc), i + window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                context_word = doc[j]
                if context_word in word_to_idx:
                    c_idx = word_to_idx[context_word]
                    cooc_matrix[w_idx, c_idx] += 1.0
                    
    # 2. Decompose Co-occurrence Matrix using SVD to get dense Word Vectors
    k_dim = min(V, 50)
    word_vectors = np.zeros((V, k_dim))
    if V > 1:
        try:
            U, S, Vt = np.linalg.svd(cooc_matrix, full_matrices=False)
            word_vectors = U[:, :k_dim] @ np.diag(S[:k_dim])
        except Exception:
            word_vectors = np.eye(V)[:, :k_dim]
    else:
        word_vectors = np.ones((V, k_dim))
        
    # 3. Compute TF-IDF weights for each word to scale word importance
    N = len(documents)
    idf = {}
    for word in vocab:
        df = sum(1 for doc in documents if word in doc)
        idf[word] = math.log(1.0 + (N / (1.0 + df)))
        
    def get_doc_vector(words):
        vec = np.zeros(k_dim)
        counts = collections.Counter(words)
        doc_len = len(words) if len(words) > 0 else 1
        for word, count in counts.items():
            if word in word_to_idx:
                w_idx = word_to_idx[word]
                tf = count / doc_len
                tfidf = tf * idf[word]
                vec += tfidf * word_vectors[w_idx]
        return vec
        
    # 4. Represent communities as weighted average vectors
    doc_vectors = []
    for doc in documents:
        doc_vectors.append(get_doc_vector(doc))
        
    # 5. Represent query as weighted average vector
    query_vector = get_doc_vector(query)
    
    # 6. Calculate Cosine Similarity
    scores = []
    q_norm = np.linalg.norm(query_vector)
    for doc_idx, cid in enumerate(cids):
        d_vec = doc_vectors[doc_idx]
        d_norm = np.linalg.norm(d_vec)
        if q_norm > 0 and d_norm > 0:
            sim = float(np.dot(query_vector, d_vec) / (q_norm * d_norm))
        else:
            sim = 0.0
        scores.append((cid, sim))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def select_relevant_communities(question: str, community_embeddings: dict[int, list[float]], community_summaries: dict[int, str], top_k: int = 3) -> list[tuple[int, float]]:
    # Use local Co-occurrence SVD + TF-IDF weighted average as the sole vector matching technique
    return select_relevant_communities_cooc_svd(question, community_summaries, top_k)

def extract_entities_from_question(question: str) -> list[str]:
    prompt = f"""Extract key entity names from the following question as a JSON list of strings.
Question: {question}

Return ONLY a valid JSON list. Example: ["entity1", "entity2"]
"""
    response = llm_client.generate_text(prompt, json_mode=True)
    try:
        from app.services.unstructured.graph_builder import parse_json_from_llm
        entities = parse_json_from_llm(response)
        if isinstance(entities, list):
            return [str(e).strip() for e in entities]
    except Exception:
        pass
    return []

def build_node_descriptions(G: nx.Graph, node_ids: list) -> dict[str, str]:
    descriptions = {}
    for nid in node_ids:
        nd = G.nodes[nid]
        desc = nd.get("description", "")
        if not desc:
            desc = f"Node {nid} of type {nd.get('type', 'General')} with label {nd.get('label', nid)} (community {nd.get('community_id', 0)})"
        descriptions[nid] = desc
    return descriptions

def compute_node_embeddings(G: nx.Graph, node_ids: list, cached_embeddings: dict) -> dict:
    missing_nids = [nid for nid in node_ids if nid not in cached_embeddings]
    if missing_nids:
        descs = build_node_descriptions(G, missing_nids)
        texts_to_embed = [descs[nid] for nid in missing_nids]
        try:
            embs = llm_client.embed_texts(texts_to_embed)
            for nid, emb in zip(missing_nids, embs):
                if emb:
                    cached_embeddings[nid] = emb
        except Exception:
            pass
    return cached_embeddings

def get_anchor_nodes_for_community(G: nx.Graph, community_id: int, question_embedding, node_embeddings: dict, entities: list[str], top_m: int = 10) -> list:
    comm_nodes = [n for n, data in G.nodes(data=True) if data.get("community_id") == community_id]
    if not comm_nodes:
        return []
        
    entity_anchors = set()
    for nid in comm_nodes:
        label = G.nodes[nid].get("label", "").lower()
        node_id = str(nid).lower()
        for ent in entities:
            ent_lower = ent.lower()
            if ent_lower == label or ent_lower == node_id or ent_lower in label:
                entity_anchors.add(nid)
                
    sim_scores = []
    if question_embedding:
        for nid in comm_nodes:
            emb = node_embeddings.get(nid, [])
            if emb:
                score = cosine_similarity(question_embedding, emb)
                sim_scores.append((nid, score))
            else:
                sim_scores.append((nid, 0.0))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        similarity_anchors = set([nid for nid, s in sim_scores[:top_m]])
    else:
        similarity_anchors = set()
        
    anchors = list(entity_anchors.union(similarity_anchors))
    if not anchors and comm_nodes:
        anchors = comm_nodes[:3]
    return anchors

def build_local_subgraph(G: nx.Graph, anchor_nodes: list, radius: int = 2) -> nx.Graph:
    subgraph_nodes = set()
    for anchor in anchor_nodes:
        ego = nx.ego_graph(G, anchor, radius=radius)
        subgraph_nodes.update(ego.nodes)
        
    return G.subgraph(subgraph_nodes)

def filtered_subgraph_to_text(G_sub: nx.Graph, anchors: list) -> str:
    nodes_lines = []
    anchor_set = set(anchors)
    for node in anchors:
        if node in G_sub.nodes:
            data = G_sub.nodes[node]
            label = data.get("label", node)
            ntype = data.get("type", "General")
            cid = data.get("community_id", 0)
            desc = data.get("description", "")
            nodes_lines.append(f"Node: ID={node}, Label={label}, Type={ntype}, Community={cid}, Description={desc}")
            
    edges_lines = []
    for u, v, data in G_sub.edges(data=True):
        if u in anchor_set or v in anchor_set:
            u_label = G_sub.nodes[u].get("label", u)
            v_label = G_sub.nodes[v].get("label", v)
            rel = data.get("relation_type", "associated_with")
            edges_lines.append(f"{u_label} --[{rel}]--> {v_label}")
            
    return "Nodes:\n" + "\n".join(nodes_lines) + "\n\nEdges:\n" + "\n".join(edges_lines)

def local_subgraph_to_text(G_sub: nx.Graph) -> str:
    nodes_lines = []
    for node, data in G_sub.nodes(data=True):
        label = data.get("label", node)
        ntype = data.get("type", "General")
        cid = data.get("community_id", 0)
        desc = data.get("description", "")
        docs = data.get("doc_ids", [])
        nodes_lines.append(f"Node: ID={node}, Label={label}, Type={ntype}, Community={cid}, Description={desc}, Source Docs={docs}")
        
    edges_lines = []
    for u, v, data in G_sub.edges(data=True):
        u_label = G_sub.nodes[u].get("label", u)
        v_label = G_sub.nodes[v].get("label", v)
        rel = data.get("relation_type", "associated_with")
        edges_lines.append(f"{u_label} --[{rel}]--> {v_label}")
        
    return "Nodes:\n" + "\n".join(nodes_lines) + "\n\nEdges:\n" + "\n".join(edges_lines)

def answer_per_community(question: str, community_id: int, community_summary: str, local_context: str) -> str:
    system_instruction = "You are a helpful assistant answering questions using a knowledge graph."
    prompt = f"""{system_instruction}

Question: {question}
Community ID: {community_id}
Community Summary:
{community_summary}

Local Graph Context:
{local_context}

Based ONLY on the community summary and local context above, provide a partial answer to the question. If the information is not relevant, say 'no information'. Be specific and list entities where possible.
"""
    return llm_client.generate_text(prompt)

def merge_answers(question: str, ranked_communities: list[tuple[int, float]], partial_answers: dict[int, str], community_summaries: dict[int, str]) -> str:
    partials_str = []
    for cid, score in ranked_communities:
        ans = partial_answers.get(cid, "no information")
        partials_str.append(f"--- Community {cid} (Similarity: {score:.4f}) ---\nAnswer: {ans}\n")
        
    prompt = f"""You are an expert synthesis engine. Your task is to merge several partial answers retrieved from different semantic communities in a knowledge graph into a single, cohesive, and comprehensive final answer to the user's question.

SYSTEM INSTRUCTIONS:
- Do not hallucinate facts. If the information is not present in the graph context, state that clearly.
- Be factual, concise, and direct in your responses.
- Do not refer to yourself as an AI or language model.

User Question: {question}

Partial Answers from Communities:
{"".join(partials_str)}

Instructions:
1. Synthesize the partial answers into a concise, direct, and well-structured final response.
2. Answer the question directly. Do NOT include redundant meta-commentary or filler descriptions, such as "there is no conflicting evidence", "this is consistently supported", or explicitly listing the node/entity types (e.g. do NOT write: "The relevant entities involved are Andrew Ng, a person, and MIT, an organization").
3. Only note contradictions if there is actual conflicting evidence in the partial answers.
4. Keep the response professional, clean, and concise.
5. CRITICAL: Do NOT mention internal technical terms such as 'communities', 'knowledge graph', 'subgraphs', 'nodes', 'edges', 'partial answers', or community IDs (e.g., 'Community 0', 'Community 1') in your response. The response must read like a direct, natural answer to the user's question.
"""
    return llm_client.generate_text(prompt)

def run_full_query_pipeline(question: str, G: nx.Graph, community_summaries: dict[int, str], community_embeddings: dict[int, list[float]], node_embeddings_cache: dict) -> dict:
    ranked_comms = select_relevant_communities(question, community_embeddings, community_summaries, top_k=3)
    entities = extract_entities_from_question(question)
    
    q_emb = []
    try:
        q_emb_list = llm_client.embed_texts([question])
        q_emb = q_emb_list[0] if q_emb_list else []
    except Exception:
        pass
        
    per_community_contexts = {}
    all_selected_node_ids = []
    for cid, score in ranked_comms:
        comm_nodes = [n for n, data in G.nodes(data=True) if data.get("community_id") == cid]
        all_selected_node_ids.extend(comm_nodes)
        
    try:
        node_embeddings_cache = compute_node_embeddings(G, all_selected_node_ids, node_embeddings_cache)
    except Exception:
        pass
        
    for cid, score in ranked_comms:
        anchors = get_anchor_nodes_for_community(G, cid, q_emb, node_embeddings_cache, entities)
        l_sub = build_local_subgraph(G, anchors, radius=2)
        local_text = local_subgraph_to_text(l_sub)
        filtered_local_text = filtered_subgraph_to_text(l_sub, anchors)
        
        summary = community_summaries.get(cid, "")
        try:
            partial_ans = answer_per_community(question, cid, summary, local_text)
        except Exception as e:
            print(f"[Query Error] answer_per_community failed for community {cid}: {e}")
            partial_ans = f"No answer (API error: {e})"
        
        per_community_contexts[cid] = {
            "summary": summary,
            "local_context": local_text,
            "filtered_local_context": filtered_local_text,
            "partial_answer": partial_ans,
            "nodes_count": len(l_sub.nodes),
            "edges_count": len(l_sub.edges),
            "anchors": anchors
        }
        
    try:
        final_ans = merge_answers(question, ranked_comms, {cid: ctx["partial_answer"] for cid, ctx in per_community_contexts.items()}, community_summaries)
    except Exception as e:
        print(f"[Query Error] merge_answers failed: {e}")
        final_ans = f"Sorry, could not generate the final answer due to an API error: {e}. Please check your API key, billing status, or model availability."
    
    validation_results = {}
    try:
        gt_parts = []
        for cid, ctx in per_community_contexts.items():
            summary_text = community_summaries.get(cid, "")
            local_context_text = ctx.get("filtered_local_context", "")
            gt_parts.append(
                f"Community {cid} Summary:\n{summary_text}\n\n"
                f"Community {cid} Subgraph Context:\n{local_context_text}"
            )
        ground_truth = "\n\n".join(gt_parts)
        from app.services.unstructured.validator import FaithfulnessValidator
        validator = FaithfulnessValidator()
        validation_results = validator.evaluate(ground_truth, final_ans)
    except Exception as e:
        print(f"[Validation Error] Faithfulness validation failed: {e}")
        
    return {
        "final_answer": final_ans,
        "ranked_communities": ranked_comms,
        "per_community_contexts": per_community_contexts,
        "extracted_entities": entities,
        "updated_node_embeddings_cache": node_embeddings_cache,
        "combined_rules": [],
        "validation_results": validation_results
    }
