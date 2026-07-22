import tempfile
import os
import networkx as nx
from pyvis.network import Network

def graph_to_pyvis(G: nx.Graph, height: str = "600px", width: str = "100%") -> str:
    # Initialize pyvis network with dark background
    net = Network(height=height, width=width, bgcolor="#0d0e15", font_color="#ffffff")
    
    # Calculate degree of each node to dynamically scale sizes
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    
    communities_in_graph = set(nx.get_node_attributes(G, 'community_id').values())
    
    # Harmonious premium color palette suited for a dark UI
    colors = [
        "#6366F1",  # Sleek Indigo
        "#60EFFF",  # Cyan Blue
        "#A855F7",  # Purple Amethyst
        "#F43F5E",  # Rose Coral
        "#F59E0B",  # Amber gold
        "#3B82F6",  # Bright Blue
        "#10B981",  # Soft Emerald
        "#EC4899",  # Soft Pink
        "#EF4444",  # Coral Red
        "#84CC16"   # Lime Green
    ]
    community_colors = {cid: colors[i % len(colors)] for i, cid in enumerate(communities_in_graph)}
    
    for node, attrs in G.nodes(data=True):
        label = attrs.get("label", node)
        ntype = attrs.get("type", "General")
        cid = attrs.get("community_id", 0)
        desc = attrs.get("description", "")
        
        title = f"ID: {node}\nLabel: {label}\nType: {ntype}\nCommunity: {cid}"
        if desc:
            title += f"\nDescription: {desc[:200]}..."
            
        color = community_colors.get(cid, "#60EFFF")
        
        # Scale sizes dynamically: base size 16, up to 40 for highly connected hubs
        deg = degrees.get(node, 1)
        size = int(16 + (deg / max_deg) * 24)
        
        net.add_node(
            node, 
            label=label, 
            title=title, 
            color={
                "background": color,
                "border": "#050608",
                "highlight": {
                    "background": "#ffffff",
                    "border": color
                }
            }, 
            group=cid,
            size=size
        )
        
    for u, v, data in G.edges(data=True):
        rel = data.get("relation_type", "associated_with")
        net.add_edge(
            u, 
            v, 
            title=rel,
            color={
                "color": "rgba(255, 255, 255, 0.22)",
                "highlight": "#6366F1",
                "hover": "#60EFFF"
            },
            width=2,
            smooth={"type": "continuous"}
        )
        
    net.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 3,
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.5)",
          "size": 8,
          "x": 2,
          "y": 2
        },
        "font": {
          "color": "#f8fafc",
          "size": 13,
          "face": "Outfit, Inter, system-ui"
        },
        "shape": "dot"
      },
      "edges": {
        "shadow": {
          "enabled": false
        },
        "font": {
          "color": "#94a3b8",
          "size": 10,
          "face": "Inter",
          "strokeWidth": 0
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": false,
        "navigationButtons": true,
        "keyboard": true
      },
      "physics": {
        "solver": "barnesHut",
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springStrength": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.5
        },
        "stabilization": {
          "enabled": true,
          "iterations": 100,
          "updateInterval": 25
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
            
        # Post-process the generated HTML to remove borders and hide the loading bar
        html_code = html_code.replace("border: 1px solid lightgray;", "border: none;")
        
        # Inject physics freeze on stabilizationFinished and auto-center
        target_instantiation = "network = new vis.Network(container, data, options);"
        replacement_instantiation = (
            "network = new vis.Network(container, data, options);\n"
            "                  network.on('stabilizationFinished', function () {\n"
            "                      network.fit();\n"
            "                      network.setOptions({ physics: false });\n"
            "                  });"
        )
        html_code = html_code.replace(target_instantiation, replacement_instantiation)
        
        custom_style = """
        <style>
        .vis-network {
            outline: none;
        }
        div.vis-network-loadingbar {
            display: none !important;
        }
        div.vis-loading {
            display: none !important;
        }
        body {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            background-color: #050608 !important;
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
