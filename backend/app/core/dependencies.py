from __future__ import annotations

from app.core.config import get_settings
from app.services.analysis_pipeline import AnalysisPipeline
from app.services.chat_service import ChatService
from app.services.codegraph_service import CodeGraphService
from app.services.graph_retrieval_service import GraphRetrievalService
from app.services.graphify_service import GraphifyService
from app.services.llm.gemini import GeminiProvider
from app.services.repo_service import RepoService
from app.services.storage import LocalStorage
from app.services.token_service import TokenService
from app.services.tree_sitter_service import TreeSitterService

settings = get_settings()
storage = LocalStorage(settings.data_dir)
token_service = TokenService()
tree_sitter_service = TreeSitterService()
codegraph_service = CodeGraphService()
graphify_service = GraphifyService(
    storage=storage,
    timeout_seconds=settings.graphify_timeout_seconds,
)
analysis_pipeline = AnalysisPipeline(
    storage=storage,
    tree_sitter_service=tree_sitter_service,
    codegraph_service=codegraph_service,
    graphify_service=graphify_service,
    token_service=token_service,
)
repo_service = RepoService(
    storage=storage,
    analysis_pipeline=analysis_pipeline,
    max_upload_mb=settings.max_upload_mb,
)
llm_provider = GeminiProvider(api_key=settings.gemini_api_key, model=settings.gemini_model)
graph_retrieval_service = GraphRetrievalService(storage=storage, token_service=token_service)
chat_service = ChatService(
    storage=storage,
    graph_retrieval_service=graph_retrieval_service,
    token_service=token_service,
    llm_provider=llm_provider,
    pipeline=analysis_pipeline,
)


