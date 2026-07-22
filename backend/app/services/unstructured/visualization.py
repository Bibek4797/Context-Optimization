import tempfile
import os
import networkx as nx
from pyvis.network import Network


# Node type → color palette for CodeGraph/Graphify
_NODE_TYPE_COLORS = {
    "module":          "#3B82F6",   # Blue  – file/module
    "file":            "#3B82F6",   # Blue  – graphify file
    "class":           "#A855F7",   # Purple – class
    "component":       "#A855F7",   # Purple – graphify component
    "function":        "#60EFFF",   # Cyan  – function
    "method":          "#34D399",   # Emerald – method
    "import":          "#94A3B8",   # Slate – imports
    "external_symbol": "#64748B",   # Grey  – external symbols
    "concept":         "#F59E0B",   # Amber – graphify concept
    "cli_output":      "#EF4444",   # Red   – cli result node
}

# Community-based palette (when community_id is meaningful, e.g. LangGraph)
_COMMUNITY_COLORS = [
    "#6366F1",  # Indigo
    "#60EFFF",  # Cyan
    "#A855F7",  # Amethyst
    "#F43F5E",  # Rose
    "#F59E0B",  # Amber
    "#3B82F6",  # Blue
    "#10B981",  # Emerald
    "#EC4899",  # Pink
    "#EF4444",  # Coral
    "#84CC16",  # Lime
]


def graph_to_pyvis(G: nx.Graph, height: str = "600px", width: str = "100%") -> str:
    net = Network(height=height, width=width, bgcolor="#0d0e15", font_color="#ffffff")

    # Degree map for node sizing
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees and max(degrees.values()) > 0 else 1

    # Community set (used for LangGraph coloring)
    communities_in_graph = set(nx.get_node_attributes(G, "community_id").values())
    has_communities = len(communities_in_graph) > 1
    community_colors = {
        cid: _COMMUNITY_COLORS[i % len(_COMMUNITY_COLORS)]
        for i, cid in enumerate(sorted(communities_in_graph))
    }

    for node, attrs in G.nodes(data=True):
        label = attrs.get("label", str(node))
        ntype = attrs.get("type", "")
        cid = attrs.get("community_id", 0)
        desc = attrs.get("description", "")

        title = f"<b>{label}</b><br/>Type: <i>{ntype}</i>"
        if cid:
            title += f"<br/>Community: {cid}"
        if desc:
            snippet = desc[:300].replace("<", "&lt;").replace(">", "&gt;")
            title += f"<br/><pre style='font-size:11px;max-width:360px;white-space:pre-wrap;color:#94a3b8'>{snippet}</pre>"

        # Color: prefer node-type color for code graphs; fall back to community color for LangGraph
        if ntype in _NODE_TYPE_COLORS:
            color = _NODE_TYPE_COLORS[ntype]
        elif has_communities:
            color = community_colors.get(cid, _COMMUNITY_COLORS[0])
        else:
            color = _COMMUNITY_COLORS[0]

        deg = degrees.get(node, 1)
        size = int(14 + (deg / max_deg) * 22)  # 14–36 px

        net.add_node(
            node,
            label=label,
            title=title,
            color={
                "background": color,
                "border": "#050608",
                "highlight": {"background": "#ffffff", "border": color},
                "hover": {"background": color, "border": "#ffffff"},
            },
            group=cid,
            size=size,
        )

    for u, v, data in G.edges(data=True):
        rel = data.get("relation_type", "")
        net.add_edge(
            u, v,
            title=rel,
            label=rel if rel else None,
            color={"color": "rgba(255,255,255,0.18)", "highlight": "#6366F1", "hover": "#60EFFF"},
            width=1.5,
            smooth={"type": "continuous"},
            font={"size": 9, "color": "#94a3b8", "strokeWidth": 0},
        )

    net.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 3,
        "shadow": { "enabled": true, "color": "rgba(0,0,0,0.55)", "size": 10, "x": 2, "y": 2 },
        "font": { "color": "#f8fafc", "size": 12, "face": "Outfit, Inter, system-ui" },
        "shape": "dot"
      },
      "edges": {
        "shadow": { "enabled": false },
        "font": { "color": "#94a3b8", "size": 9, "face": "Inter", "strokeWidth": 0 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 150,
        "hideEdgesOnDrag": true,
        "navigationButtons": true,
        "keyboard": true,
        "zoomView": true
      },
      "physics": {
        "solver": "barnesHut",
        "barnesHut": {
          "gravitationalConstant": -4500,
          "centralGravity": 0.4,
          "springLength": 120,
          "springStrength": 0.06,
          "damping": 0.45,
          "avoidOverlap": 0.8
        },
        "stabilization": {
          "enabled": true,
          "iterations": 2000,
          "updateInterval": 50,
          "fit": true
        }
      }
    }
    """)

    fd, path = tempfile.mkstemp(suffix=".html")
    try:
        os.close(fd)
        net.write_html(path)
        with open(path, "r", encoding="utf-8") as f:
            html_code = f.read()

        # Remove default border
        html_code = html_code.replace("border: 1px solid lightgray;", "border: none;")

        # Disable physics the moment stabilization finishes, then smooth-fit the view
        target_instantiation = "network = new vis.Network(container, data, options);"
        replacement_instantiation = (
            "network = new vis.Network(container, data, options);\n"
            "                  network.once('stabilizationIterationsDone', function () {\n"
            "                      network.setOptions({ physics: { enabled: false } });\n"
            "                      network.fit({ animation: { duration: 400, easingFunction: 'easeInOutQuad' } });\n"
            "                  });\n"
            "                  network.on('stabilizationProgress', function(params) {\n"
            "                      if (params.iterations >= params.total) {\n"
            "                          network.setOptions({ physics: { enabled: false } });\n"
            "                      }\n"
            "                  });"
        )
        html_code = html_code.replace(target_instantiation, replacement_instantiation)

        custom_style = """
        <style>
        .vis-network { outline: none; }
        div.vis-network-loadingbar { display: none !important; }
        div.vis-loading { display: none !important; }
        body {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            background-color: #0d0e15 !important;
        }
        </style>
        """
        html_code = html_code.replace("</head>", f"{custom_style}</head>")

    finally:
        try:
            os.remove(path)
        except Exception:
            pass

    return html_code


def ego_graph_to_pyvis(G: nx.Graph, center_node, radius: int = 2, height: str = "400px", width: str = "100%") -> str:
    ego_G = nx.ego_graph(G, center_node, radius=radius)
    return graph_to_pyvis(ego_G, height=height, width=width)
