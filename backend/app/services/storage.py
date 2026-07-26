from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models.schemas import (
    CodeChunk,
    GraphDocument,
    QueryRecord,
    RepoFile,
    RepoMetadata,
    TokenSummary,
    TreeSitterDocument,
)


class LocalStorage:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.repos_dir = data_dir / "repos"
        self.uploads_dir = data_dir / "uploads"
        self.state_dir = data_dir / "state"
        self.logs_dir = data_dir / "logs"
        for path in [self.repos_dir, self.uploads_dir, self.state_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def repo_source_dir(self, repo_id: str) -> Path:
        return self.repos_dir / repo_id / "source"

    def repo_state_dir(self, repo_id: str) -> Path:
        path = self.state_dir / "repos" / repo_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_repo_metadata(self, metadata: RepoMetadata) -> None:
        metadata.updated_at = datetime.now(timezone.utc)
        self._save_json(self.repo_state_dir(metadata.repo_id) / "metadata.json", metadata.model_dump(mode="json"))

    def load_repo_metadata(self, repo_id: str) -> RepoMetadata | None:
        data = self._load_json(self.repo_state_dir(repo_id) / "metadata.json")
        return RepoMetadata.model_validate(data) if data else None

    def list_repos(self) -> list[RepoMetadata]:
        repos_dir = self.state_dir / "repos"
        if not repos_dir.exists():
            return []
        metas = []
        for p in repos_dir.iterdir():
            if p.is_dir():
                meta = self.load_repo_metadata(p.name)
                if meta:
                    metas.append(meta)
        metas.sort(key=lambda m: getattr(m, "created_at", datetime.now(timezone.utc)), reverse=True)
        return metas

    def save_files(self, repo_id: str, files: list[RepoFile]) -> None:
        self._save_json(self.repo_state_dir(repo_id) / "files.json", [file.model_dump(mode="json") for file in files])

    def load_files(self, repo_id: str) -> list[RepoFile]:
        data = self._load_json(self.repo_state_dir(repo_id) / "files.json") or []
        return [RepoFile.model_validate(item) for item in data]

    def save_tree_sitter(self, repo_id: str, file_path: str, document: TreeSitterDocument) -> None:
        path = self.repo_state_dir(repo_id) / "tree_sitter" / f"{self._path_key(file_path)}.json"
        self._save_json(path, document.model_dump(mode="json"))

    def load_tree_sitter(self, repo_id: str, file_path: str) -> TreeSitterDocument | None:
        path = self.repo_state_dir(repo_id) / "tree_sitter" / f"{self._path_key(file_path)}.json"
        data = self._load_json(path)
        return TreeSitterDocument.model_validate(data) if data else None

    def save_codegraph(self, repo_id: str, graph: GraphDocument) -> None:
        self._save_json(self.repo_state_dir(repo_id) / "codegraph.json", graph.model_dump(mode="json"))

    def load_codegraph(self, repo_id: str) -> GraphDocument | None:
        data = self._load_json(self.repo_state_dir(repo_id) / "codegraph.json")
        return GraphDocument.model_validate(data) if data else None

    def save_graphify(self, repo_id: str, graph: GraphDocument) -> None:
        self._save_json(self.repo_state_dir(repo_id) / "graphify.json", graph.model_dump(mode="json"))

    def load_graphify(self, repo_id: str) -> GraphDocument | None:
        data = self._load_json(self.repo_state_dir(repo_id) / "graphify.json")
        return GraphDocument.model_validate(data) if data else None

    def save_chunks(self, repo_id: str, chunks: list[CodeChunk]) -> None:
        self._save_json(self.repo_state_dir(repo_id) / "chunks.json", [chunk.model_dump(mode="json") for chunk in chunks])

    def load_chunks(self, repo_id: str) -> list[CodeChunk]:
        data = self._load_json(self.repo_state_dir(repo_id) / "chunks.json") or []
        return [CodeChunk.model_validate(item) for item in data]

    def save_token_summary(self, repo_id: str, summary: TokenSummary) -> None:
        self._save_json(self.repo_state_dir(repo_id) / "token_summary.json", summary.model_dump(mode="json"))

    def load_token_summary(self, repo_id: str) -> TokenSummary | None:
        data = self._load_json(self.repo_state_dir(repo_id) / "token_summary.json")
        return TokenSummary.model_validate(data) if data else None

    def save_query(self, record: QueryRecord) -> None:
        repo_queries = self.repo_state_dir(record.repo_id) / "queries"
        self._save_json(repo_queries / f"{record.query_id}.json", record.model_dump(mode="json"))
        self._save_json(self.state_dir / "queries" / f"{record.query_id}.json", record.model_dump(mode="json"))
        self._update_cumulative_usage(record)

    def find_query(self, query_id: str) -> QueryRecord | None:
        data = self._load_json(self.state_dir / "queries" / f"{query_id}.json")
        if not data:
            return None
        try:
            return QueryRecord.model_validate(data)
        except Exception:
            return None

    def append_log(self, repo_id: str, stage: str, level: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        path = self.repo_state_dir(repo_id) / "logs.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "level": level,
            "message": message,
            "metadata": metadata or {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def load_logs(self, repo_id: str, limit: int = 200) -> list[dict[str, Any]]:
        path = self.repo_state_dir(repo_id) / "logs.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()]

    def _update_cumulative_usage(self, record: QueryRecord) -> None:
        try:
            summary = self.load_token_summary(record.repo_id)
            if summary is None:
                return
            cumulative = dict(summary.cumulative_session_usage)
            for key, measurement in record.token_usage.items():
                if isinstance(measurement, dict):
                    t_val = measurement.get("tokens", 0)
                else:
                    t_val = getattr(measurement, "tokens", 0)
                cumulative[key] = cumulative.get(key, 0) + int(t_val or 0)
            summary.cumulative_session_usage = cumulative
            summary.updated_at = datetime.now(timezone.utc)
            self.save_token_summary(record.repo_id, summary)
        except Exception:
            pass

    def _path_key(self, path: str) -> str:
        return hashlib.sha1(path.encode("utf-8")).hexdigest()

    def _save_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _load_json(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))

