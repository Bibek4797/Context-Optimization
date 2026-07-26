from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.models.schemas import RepoMetadata
from app.services.analysis_pipeline import AnalysisPipeline
from app.services.file_utils import clean_repo_name, safe_extract_zip, validate_github_url
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

