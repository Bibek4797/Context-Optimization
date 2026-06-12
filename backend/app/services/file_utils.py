from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from app.models.schemas import RepoFile

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
    ".vscode",
    ".next",
}

IGNORED_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "go.sum",
    "poetry.lock",
    "mix.lock",
    "composer.lock",
}


def clean_repo_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
    return cleaned or "repo"


def read_text_lossy(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def source_snippet(text: str, line_start: int, line_end: int, max_chars: int | None = 800) -> str:
    lines = text.splitlines()
    start = max(0, line_start - 1)
    end = min(len(lines), line_end)
    snippet = "\n".join(lines[start:end])
    if max_chars is not None and len(snippet) > max_chars:
        return snippet[:max_chars] + "\n..."
    return snippet


def is_ignored(path: Path) -> bool:
    if path.name in IGNORED_FILES:
        return True
    return any(part in IGNORED_DIRS for part in path.parts)


EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}


def iter_code_files(root: Path) -> list[RepoFile]:
    files: list[RepoFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if is_ignored(rel):
            continue
        suffix = path.suffix.lower()
        if suffix in EXTENSION_TO_LANGUAGE:
            text = read_text_lossy(path)
            language = EXTENSION_TO_LANGUAGE[suffix]
            files.append(
                RepoFile(
                    path=rel.as_posix(),
                    language=language,
                    size_bytes=path.stat().st_size,
                    line_count=len(text.splitlines()),
                )
            )
    return files


def iter_python_files(root: Path) -> list[RepoFile]:
    return iter_code_files(root)


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = destination / member.filename
            resolved = member_path.resolve()
            if destination_resolved not in resolved.parents and resolved != destination_resolved:
                raise ValueError(f"Unsafe zip path detected: {member.filename}")
            if is_ignored(Path(member.filename)):
                continue
            archive.extract(member, destination)
    _flatten_single_top_level_dir(destination)


def _flatten_single_top_level_dir(destination: Path) -> None:
    children = [child for child in destination.iterdir() if child.name not in {"__MACOSX"}]
    if len(children) != 1 or not children[0].is_dir():
        return
    inner = children[0]
    tmp = destination.with_name(destination.name + "-flattening")
    if tmp.exists():
        shutil.rmtree(tmp)
    inner.rename(tmp)
    shutil.rmtree(destination)
    tmp.rename(destination)


def validate_github_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("GitHub URL must use http or https.")
    if parsed.netloc.lower() != "github.com":
        raise ValueError("Stage 1 Git import only accepts github.com URLs.")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError("GitHub URL must include owner and repository.")
    owner, repo = parts[0], parts[1].removesuffix(".git")
    if not owner or not repo:
        raise ValueError("GitHub URL must include owner and repository.")
    return f"https://github.com/{owner}/{repo}.git"

