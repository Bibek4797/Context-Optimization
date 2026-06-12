from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.models.schemas import RepoMetadata, RepoStatus, RepoStats, RepoFile
from app.services.codegraph_service import CodeGraphService
from app.services.file_utils import is_ignored, iter_python_files, read_text_lossy
from app.services.graphify_service import GraphifyService
from app.services.storage import LocalStorage
from app.services.token_service import TokenService
from app.services.tree_sitter_service import TreeSitterService


class AnalysisPipeline:
    def __init__(
        self,
        storage: LocalStorage,
        tree_sitter_service: TreeSitterService,
        codegraph_service: CodeGraphService,
        graphify_service: GraphifyService,
        token_service: TokenService,
    ) -> None:
        self.storage = storage
        self.tree_sitter_service = tree_sitter_service
        self.codegraph_service = codegraph_service
        self.graphify_service = graphify_service
        self.token_service = token_service

    def analyze_existing(self, name: str, source_dir: Path, origin: str, repo_id: str | None = None) -> RepoMetadata:
        repo_id = repo_id or uuid4().hex
        metadata = RepoMetadata(repo_id=repo_id, name=name, origin=origin, status=RepoStatus.running)
        self.storage.save_repo_metadata(metadata)
        self.storage.append_log(repo_id, "pipeline", "info", "Analysis started.")

        try:
            from app.services.file_utils import iter_code_files
            files = iter_code_files(source_dir)
            stats = self._stats(source_dir, files)
            metadata.stats = stats
            if not files:
                metadata.status = RepoStatus.failed
                metadata.error = "No supported code files found. The engine supports Python, JavaScript/TypeScript, Go, Rust, Java, and C/C++ repositories."
                self.storage.save_repo_metadata(metadata)
                self.storage.append_log(repo_id, "ingestion", "error", metadata.error)
                return metadata

            self.storage.save_files(repo_id, files)
            self.storage.append_log(repo_id, "ingestion", "info", f"Discovered {len(files)} code files.")

            parsed_files = []
            warnings: list[str] = []
            for repo_file in files:
                document = self.tree_sitter_service.parse_file(repo_id, source_dir, repo_file.path, repo_file.language)
                self.storage.save_tree_sitter(repo_id, repo_file.path, document)
                if document.parse_error:
                    repo_file.parse_status = "failed"
                    repo_file.parse_error = document.parse_error
                    warnings.append(f"Tree-sitter failed for {repo_file.path}: {document.parse_error}")
                    self.storage.append_log(repo_id, "tree-sitter", "warning", warnings[-1])
                else:
                    repo_file.parse_status = "parsed"
                    parsed_files.append(repo_file.path)
            self.storage.save_files(repo_id, files)
            self.storage.append_log(repo_id, "tree-sitter", "info", f"Parsed {len(parsed_files)} files.")

            codegraph = self.codegraph_service.build(repo_id, source_dir, files)
            self.storage.save_codegraph(repo_id, codegraph)
            self.storage.append_log(
                repo_id,
                "codegraph",
                "info",
                f"Generated {len(codegraph.nodes)} nodes and {len(codegraph.edges)} edges.",
            )

            graphify = self.graphify_service.run_or_fallback(repo_id, source_dir, codegraph)
            self.storage.save_graphify(repo_id, graphify)
            for warning in graphify.warnings:
                warnings.append(warning)
                self.storage.append_log(repo_id, "graphify", "warning", warning)
            self.storage.append_log(
                repo_id,
                "graphify",
                "info",
                f"Stored {graphify.source} graph with {len(graphify.nodes)} nodes and {len(graphify.edges)} edges.",
            )

            chunks = self.token_service.build_chunks(source_dir, files)
            self.storage.save_chunks(repo_id, chunks)
            summary = self.token_service.build_repo_summary(repo_id, source_dir, files, chunks, codegraph, graphify)
            self.storage.save_token_summary(repo_id, summary)
            self.storage.append_log(repo_id, "tokens", "info", "Token summary generated.")

            metadata.warnings = warnings
            metadata.status = RepoStatus.partial if warnings else RepoStatus.completed
            self.storage.save_repo_metadata(metadata)
            self.storage.append_log(repo_id, "pipeline", "info", f"Analysis finished with status {metadata.status}.")
            return metadata
        except Exception as exc:
            metadata.status = RepoStatus.failed
            metadata.error = str(exc)
            self.storage.save_repo_metadata(metadata)
            self.storage.append_log(repo_id, "pipeline", "error", str(exc))
            return metadata

    def _stats(self, source_dir: Path, files: list[RepoFile]) -> RepoStats:
        total_files = 0
        total_lines = 0
        for path in source_dir.rglob("*"):
            if path.is_file():
                rel = path.relative_to(source_dir)
                if is_ignored(rel):
                    continue
                total_files += 1
                if path.suffix in {".py", ".pyi", ".txt", ".md", ".toml", ".yaml", ".yml", ".json"}:
                    total_lines += len(read_text_lossy(path).splitlines())
        python_lines = sum(file.line_count for file in files)
        total_tokens = sum(
            self.token_service.estimate_tokens(read_text_lossy(source_dir / file.path))
            for file in files
        )
        return RepoStats(
            total_files=total_files,
            python_files=len(files),
            total_lines=total_lines,
            python_lines=python_lines,
            total_tokens=total_tokens,
        )
