from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile

from app.services.analysis_pipeline import AnalysisPipeline
from app.services.codegraph_service import CodeGraphService
from app.services.file_utils import safe_upload_relative_path
from app.services.graphify_service import GraphifyService
from app.services.repo_service import RepoService
from app.services.storage import LocalStorage
from app.services.token_service import TokenService
from app.services.tree_sitter_service import TreeSitterService


def test_safe_upload_relative_path_rejects_traversal_and_absolute_paths() -> None:
    assert safe_upload_relative_path("sample/src/app.py").as_posix() == "sample/src/app.py"

    for unsafe_path in ("../secret.py", "/tmp/secret.py", "C:\\tmp\\secret.py"):
        with pytest.raises(ValueError):
            safe_upload_relative_path(unsafe_path)


def test_ingest_file_uploads_preserves_folder_paths(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "data")
    pipeline = AnalysisPipeline(
        storage=storage,
        tree_sitter_service=TreeSitterService(),
        codegraph_service=CodeGraphService(),
        graphify_service=GraphifyService(storage=storage, cli_name="__missing_graphify__"),
        token_service=TokenService(),
    )
    service = RepoService(storage=storage, analysis_pipeline=pipeline, max_upload_mb=1)
    uploads = [
        UploadFile(filename="app.py", file=BytesIO(b"def hello():\n    return 'hi'\n")),
        UploadFile(filename="util.py", file=BytesIO(b"from .app import hello\n\ndef call():\n    return hello()\n")),
    ]

    metadata = asyncio.run(service.ingest_file_uploads(uploads, ["sample/src/app.py", "sample/src/util.py"]))

    assert metadata.name == "sample"
    assert metadata.status == "completed"
    assert metadata.stats.python_files == 2
    assert sorted(file.path for file in storage.load_files(metadata.repo_id)) == ["src/app.py", "src/util.py"]
