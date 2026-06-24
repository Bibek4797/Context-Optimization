from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from app.services.storage import LocalStorage


def clean_xml_code_block(code: str) -> str:
    # 1. Remove leading/trailing newlines and carriage returns, keeping leading/trailing spaces
    code = code.strip("\r\n")
    
    # 2. Check for markdown code block fences
    stripped_code = code.strip()
    if stripped_code.startswith("```"):
        lines = code.splitlines()
        start_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                start_idx = i
                break
        end_idx = -1
        for i in range(len(lines) - 1, start_idx, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        
        if start_idx != -1:
            if end_idx != -1:
                code_lines = lines[start_idx+1:end_idx]
            else:
                code_lines = lines[start_idx+1:]
            code = "\n".join(code_lines)
            code = code.strip("\r\n")
            
    return code


def adjust_indentation(replacement_str: str, file_indent: str, base_indent_len: int) -> str:
    repl_lines = replacement_str.splitlines()
    if not repl_lines:
        return replacement_str
        
    delta_indent = len(file_indent) - base_indent_len
    adjusted_lines = []
    
    for line in repl_lines:
        if not line.strip():
            adjusted_lines.append("")
            continue
            
        if delta_indent > 0:
            adjusted_line = (" " * delta_indent) + line
        elif delta_indent < 0:
            remove_len = abs(delta_indent)
            line_leading_spaces = len(line) - len(line.lstrip())
            to_remove = min(remove_len, line_leading_spaces)
            adjusted_line = line[to_remove:]
        else:
            adjusted_line = line
        adjusted_lines.append(adjusted_line)
        
    return "\n".join(adjusted_lines)


class RectificationService:
    def __init__(self, storage: LocalStorage, pipeline: Any) -> None:
        self.storage = storage
        self.pipeline = pipeline

    def apply_code_fix(self, repo_id: str, file_path: str, original_code: str, replacement_code: str) -> dict[str, Any]:
        repo_root = self.storage.repo_source_dir(repo_id)
        if not repo_root.exists():
            return {"status": "failed", "error": "Repository source folder not found."}
            
        abs_path = (repo_root / file_path).resolve()
        
        # Security check: Prevent directory traversal out of the repository root
        try:
            if not abs_path.is_relative_to(repo_root.resolve()):
                return {"status": "failed", "error": "Invalid file path: path must reside inside repository root."}
        except ValueError:
            return {"status": "failed", "error": "Invalid file path structure."}

        if not abs_path.exists():
            return {"status": "failed", "error": f"File '{file_path}' does not exist on disk."}

        try:
            # Read existing file content and normalize carriage returns to standard Unix newlines
            content = abs_path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
            
            # Normalize target and replacement newlines, and clean/strip markdown fences properly
            target_str = clean_xml_code_block(original_code).replace("\r\n", "\n")
            target_replacement = clean_xml_code_block(replacement_code).replace("\r\n", "\n")
            
            new_content = None
            
            # Layer A: Exact match
            if target_str in content:
                new_content = content.replace(target_str, target_replacement, 1)
            else:
                # Layer B: Find target block ignoring leading/trailing whitespaces but preserving line structure
                target_lines = [l.strip() for l in target_str.splitlines()]
                content_lines = content.splitlines()
                
                match_idx = -1
                # Basic rolling window search for the block
                for i in range(len(content_lines) - len(target_lines) + 1):
                    window = [content_lines[i + j].strip() for j in range(len(target_lines))]
                    if window == target_lines:
                        match_idx = i
                        break
                        
                if match_idx != -1:
                    # Find base indentation of first line in file matched block
                    file_first_line = content_lines[match_idx]
                    file_indent_len = len(file_first_line) - len(file_first_line.lstrip())
                    file_indent = file_first_line[:file_indent_len]
                    
                    # Find base indentation of first line in proposed target_str
                    target_first_line = target_str.splitlines()[0] if target_str.splitlines() else ""
                    target_indent_len = len(target_first_line) - len(target_first_line.lstrip())
                    
                    # Adjust indentation of target_replacement to match the file's indentation
                    adjusted_replacement = adjust_indentation(target_replacement, file_indent, target_indent_len)
                    
                    # Reconstruct the file with the replacement
                    before = "\n".join(content_lines[:match_idx])
                    after = "\n".join(content_lines[match_idx + len(target_lines):])
                    new_content = (before + "\n" if before else "") + adjusted_replacement + ("\n" + after if after else "")
            
            if new_content is None:
                return {
                    "status": "failed", 
                    "error": (
                        "Original code block could not be located in the file. "
                        "This can happen if the block was modified previously or formatted differently."
                    )
                }
                
            # Create safety backup file
            backup_path = abs_path.with_suffix(abs_path.suffix + ".bak")
            shutil.copy2(abs_path, backup_path)
            
            # Save updated file
            abs_path.write_text(new_content, encoding="utf-8")
            
            # Re-run pipeline analysis to dynamically rebuild CodeGraph, Graphify and chunks instantly!
            metadata = self.storage.load_repo_metadata(repo_id)
            if metadata:
                self.pipeline.analyze_existing(
                    name=metadata.name, 
                    source_dir=repo_root, 
                    origin=metadata.origin, 
                    repo_id=repo_id
                )
                
            return {
                "status": "success",
                "file_path": file_path,
                "backup_path": str(backup_path.name),
                "new_content": new_content,
                "message": f"Successfully applied changes to '{file_path}'. A backup copy was created."
            }
            
        except Exception as exc:
            return {"status": "failed", "error": f"Error applying fix: {exc}"}
