from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from collections.abc import Sequence
from uuid import uuid4

from fastapi import UploadFile

from app.models.schemas import RepoMetadata
from app.services.analysis_pipeline import AnalysisPipeline
from app.services.file_utils import (
    clean_repo_name,
    flatten_single_top_level_dir,
    is_ignored,
    safe_extract_zip,
    safe_upload_relative_path,
    validate_github_url,
)
from app.services.storage import LocalStorage


class RepoService:
    def __init__(self, storage: LocalStorage, analysis_pipeline: AnalysisPipeline, max_upload_mb: int) -> None:
        self.storage = storage
        self.analysis_pipeline = analysis_pipeline
        self.max_upload_mb = max_upload_mb

    async def ingest_zip_upload(self, file: UploadFile) -> RepoMetadata:
        repo_id = uuid4().hex
        name = clean_repo_name(Path(file.filename or "repo.zip").stem)
        upload_path = self.storage.uploads_dir / f"{repo_id}.zip"
        source_dir = self.storage.repo_source_dir(repo_id)
        upload_path.parent.mkdir(parents=True, exist_ok=True)

        size = 0
        with upload_path.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > self.max_upload_mb * 1024 * 1024:
                    raise ValueError(f"Upload exceeds {self.max_upload_mb} MB limit.")
                handle.write(chunk)

        if source_dir.exists():
            shutil.rmtree(source_dir)
        safe_extract_zip(upload_path, source_dir)
        return self.analysis_pipeline.analyze_existing(name=name, source_dir=source_dir, origin="upload", repo_id=repo_id)

    async def ingest_file_uploads(self, files: Sequence[UploadFile], paths: Sequence[str] | None = None) -> RepoMetadata:
        if not files:
            raise ValueError("Upload must include at least one file.")

        if len(files) == 1 and (files[0].filename or "").lower().endswith(".zip"):
            return await self.ingest_zip_upload(files[0])
        if any((file.filename or "").lower().endswith(".zip") for file in files):
            raise ValueError("Upload a single .zip file or source files/folders, not both.")

        repo_id = uuid4().hex
        relative_paths = [
            safe_upload_relative_path(paths[index] if paths and index < len(paths) and paths[index] else file.filename or f"file-{index}")
            for index, file in enumerate(files)
        ]
        name = self._uploaded_files_repo_name(relative_paths)
        source_dir = self.storage.repo_source_dir(repo_id)
        if source_dir.exists():
            shutil.rmtree(source_dir)
        source_dir.mkdir(parents=True, exist_ok=True)

        size = 0
        written = 0
        for file, relative_path in zip(files, relative_paths, strict=True):
            if is_ignored(relative_path):
                continue
            destination = source_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_upload_mb * 1024 * 1024:
                        raise ValueError(f"Upload exceeds {self.max_upload_mb} MB limit.")
                    handle.write(chunk)
            written += 1

        if written == 0:
            raise ValueError("No supported files were saved after applying ignore rules.")

        flatten_single_top_level_dir(source_dir)
        return self.analysis_pipeline.analyze_existing(name=name, source_dir=source_dir, origin="file_upload", repo_id=repo_id)

    def import_github(self, url: str) -> RepoMetadata:
        clone_url = validate_github_url(url)
        repo_id = uuid4().hex
        name = clean_repo_name(Path(clone_url.removesuffix(".git")).name)
        source_dir = self.storage.repo_source_dir(repo_id)
        if source_dir.exists():
            shutil.rmtree(source_dir)
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, str(source_dir)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=180,
        )
        if result.returncode != 0:
            raise ValueError(f"Git clone failed: {(result.stderr or '').strip() or (result.stdout or '').strip()}")
        return self.analysis_pipeline.analyze_existing(name=name, source_dir=source_dir, origin=clone_url, repo_id=repo_id)

    def _uploaded_files_repo_name(self, paths: Sequence[Path]) -> str:
        if len(paths) == 1:
            return clean_repo_name(paths[0].stem)

        first_parts = {path.parts[0] for path in paths if len(path.parts) > 1}
        if len(first_parts) == 1:
            return clean_repo_name(next(iter(first_parts)))
        return f"uploaded-{len(paths)}-files"
