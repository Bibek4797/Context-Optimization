import io
import sys
from pathlib import Path

# Try importing tree-sitter packages
try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython
except ImportError as e:
    print(f"Error importing tree-sitter libraries: {e}")
    print("Please install them using: pip install tree-sitter tree-sitter-python")
    sys.exit(1)


def generate_tree_string(node, source_code: bytes, prefix: str = "", is_last: bool = True) -> str:
    """Recursively generate a tree structure using box-drawing characters."""
    children = node.children
    
    node_text = source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
    node_text_clean = node_text.replace("\n", " ")
    if len(node_text_clean) > 40:
        node_text_clean = node_text_clean[:37] + "..."
        
    preview = f' "{node_text_clean}"' if node.child_count == 0 and node_text_clean else ""
    marker = "└── " if is_last else "├── "
    
    line = f"{prefix}{marker}{node.type} [{node.start_point[0]}:{node.start_point[1]} - {node.end_point[0]}:{node.end_point[1]}]{preview}\n"
    new_prefix = prefix + ("    " if is_last else "│   ")
    
    result = line
    for i, child in enumerate(children):
        child_is_last = (i == len(children) - 1)
        result += generate_tree_string(child, source_code, new_prefix, child_is_last)
    return result


def parse_and_format(code_content: str) -> str:
    """Parse python code content and return the formatted CST tree."""
    try:
        py_lang = Language(tspython.language())
        parser = Parser()
        parser.language = py_lang
    except Exception as e:
        try:
            parser = Parser()
            parser.set_language(Language(tspython.language()))
        except Exception as e2:
            return f"Error setting language parser: {e}\nFallback error: {e2}"
            
    source_bytes = code_content.encode("utf-8")
    tree = parser.parse(source_bytes)
    
    root = tree.root_node
    node_text = source_bytes[root.start_byte:root.end_byte].decode("utf-8", errors="replace").strip()
    node_text_clean = node_text.replace("\n", " ")
    if len(node_text_clean) > 40:
        node_text_clean = node_text_clean[:37] + "..."
    preview = f' "{node_text_clean}"' if root.child_count == 0 and node_text_clean else ""
    
    result = f"{root.type} [{root.start_point[0]}:{root.start_point[1]} - {root.end_point[0]}:{root.end_point[1]}]{preview}\n"
    
    children = root.children
    for i, child in enumerate(children):
        child_is_last = (i == len(children) - 1)
        result += generate_tree_string(child, source_bytes, "", child_is_last)
    return result


def is_streamlit_running() -> bool:
    try:
        from streamlit.runtime import exists
        return exists()
    except ImportError:
        return False


def safe_print(text: str):
    """Print with fallback to ASCII if the console encoding doesn't support UTF-8 characters."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback to plain ASCII characters if console doesn't support UTF-8
        ascii_text = text.replace("└── ", "+-- ").replace("├── ", "+-- ").replace("│   ", "|   ")
        print(ascii_text)


if __name__ == "__main__":
    # Streamlit Mode
    if is_streamlit_running():
        import streamlit as st
        st.set_page_config(page_title="Tree-sitter Parser", layout="wide")
        
        st.title("🌲 Tree-sitter Parser")
        
        mode = st.radio("Input Mode", ["Paste Code", "Upload File"], horizontal=True)
        
        code_input = ""
        if mode == "Paste Code":
            code_input = st.text_area(
                "Paste your Python code here:",
                value="""def greet(name):\n    print(f"Hello, {name}!")\n\ngreet("World")""",
                height=200
            )
        else:
            uploaded_file = st.file_uploader("Upload a Python file (.py)", type=["py"])
            if uploaded_file is not None:
                code_input = uploaded_file.getvalue().decode("utf-8", errors="replace")
                
        if code_input:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📄 Source Code")
                st.code(code_input, language="python")
            with col2:
                st.subheader("🌲 Tree Output")
                tree_str = parse_and_format(code_input)
                st.text_area("Tree Hierarchy", tree_str, height=500)
    
    # CLI Mode
    else:
        args = sys.argv[1:]
        
        # Ensure stdout handles UTF-8 on Windows CLI if supported
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

        if len(args) > 0:
            file_path = Path(args[0])
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    safe_print(f"--- Tree for {file_path.name} ---")
                    safe_print(parse_and_format(content))
                except Exception as e:
                    safe_print(f"Error reading file: {e}")
            else:
                safe_print(f"File not found: {file_path}")
        else:
            safe_print("Usage: python tree_sitter_demo.py <path_to_file.py>")
            safe_print("Or run as UI app: streamlit run tree_sitter_demo.py\n")
            safe_print("Running built-in example:")
            sample = 'def greet(name):\n    print(f"Hello, {name}!")\n'
            safe_print(f"--- Source ---\n{sample}\n--- Tree ---")
            safe_print(parse_and_format(sample))
