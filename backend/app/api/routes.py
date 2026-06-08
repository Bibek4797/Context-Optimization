from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.dependencies import chat_service, repo_service, storage
from app.models.schemas import (
    ChatRequest,
    GitHubImportRequest,
    QueryRecord,
    RepoFileList,
    RepoMetadata,
    TreeSitterDocument,
)

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/repo/upload", response_model=RepoMetadata)
async def upload_repo(file: UploadFile = File(...)) -> RepoMetadata:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload must be a .zip file.")
    try:
        return await repo_service.ingest_zip_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/repo/import-github", response_model=RepoMetadata)
def import_github(request: GitHubImportRequest) -> RepoMetadata:
    try:
        return repo_service.import_github(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/repo/{repo_id}/status", response_model=RepoMetadata)
def repo_status(repo_id: str) -> RepoMetadata:
    metadata = storage.load_repo_metadata(repo_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Repo not found.")
    return metadata


@router.get("/repo/{repo_id}/files", response_model=RepoFileList)
def repo_files(repo_id: str) -> RepoFileList:
    metadata = storage.load_repo_metadata(repo_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Repo not found.")
    return RepoFileList(repo_id=repo_id, files=storage.load_files(repo_id))


@router.get("/repo/{repo_id}/tree-sitter", response_model=TreeSitterDocument)
def tree_sitter(repo_id: str, file: str = Query(..., min_length=1)) -> TreeSitterDocument:
    document = storage.load_tree_sitter(repo_id, file)
    if document is None:
        raise HTTPException(status_code=404, detail="Tree-sitter output not found for file.")
    return document


@router.get("/repo/{repo_id}/codegraph")
def codegraph(repo_id: str):
    graph = storage.load_codegraph(repo_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="CodeGraph output not found.")
    return graph


@router.get("/repo/{repo_id}/graphify")
def graphify(repo_id: str):
    graph = storage.load_graphify(repo_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Graphify output not found.")
    return graph


@router.get("/repo/{repo_id}/token-summary")
def token_summary(repo_id: str):
    summary = storage.load_token_summary(repo_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Token summary not found.")
    return summary


@router.get("/repo/{repo_id}/logs")
def repo_logs(repo_id: str, limit: int = Query(200, ge=1, le=1000)):
    metadata = storage.load_repo_metadata(repo_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Repo not found.")
    return {"repo_id": repo_id, "logs": storage.load_logs(repo_id, limit=limit)}
@router.post("/chat/graph-optimized", response_model=QueryRecord)
def graph_optimized_chat(request: ChatRequest) -> QueryRecord:
    try:
        return chat_service.graph_optimized_qa(request.repo_id, request.query, request.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/query/{query_id}/details", response_model=QueryRecord)
def query_details(query_id: str) -> QueryRecord:
    record = storage.find_query(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Query not found.")
    return record

