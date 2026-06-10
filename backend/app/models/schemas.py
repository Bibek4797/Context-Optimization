from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RepoStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    partial = "partial"
    failed = "failed"


class CountType(str, Enum):
    exact = "exact"
    estimated = "estimated"


class RepoStats(BaseModel):
    total_files: int = 0
    python_files: int = 0
    total_lines: int = 0
    python_lines: int = 0


class RepoMetadata(BaseModel):
    repo_id: str
    name: str
    origin: str
    status: RepoStatus = RepoStatus.pending
    stats: RepoStats = Field(default_factory=RepoStats)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RepoFile(BaseModel):
    path: str
    language: str = "python"
    size_bytes: int
    line_count: int
    parse_status: Literal["pending", "parsed", "failed"] = "pending"
    parse_error: str | None = None


class RepoFileList(BaseModel):
    repo_id: str
    files: list[RepoFile]


class TreeNode(BaseModel):
    type: str
    named: bool
    start_point: tuple[int, int]
    end_point: tuple[int, int]
    start_byte: int
    end_byte: int
    text_preview: str | None = None
    children: list["TreeNode"] = Field(default_factory=list)


class TreeSitterDocument(BaseModel):
    repo_id: str
    file_path: str
    language: str = "python"
    root: TreeNode | None = None
    source: str
    warnings: list[str] = Field(default_factory=list)
    parse_error: str | None = None
    generated_at: datetime = Field(default_factory=utc_now)


class GraphNode(BaseModel):
    node_id: str
    node_type: str
    label: str
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    source_snippet: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    edge_id: str
    edge_type: str
    source_node: str
    target_node: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphDocument(BaseModel):
    repo_id: str
    source: Literal["codegraph", "graphify", "graphify-fallback"]
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    raw_output_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)


class TokenMeasurement(BaseModel):
    stage: str
    tokens: int
    count_type: CountType
    provider: str | None = None
    model: str | None = None
    notes: str | None = None


class TokenSummary(BaseModel):
    repo_id: str
    stages: dict[str, TokenMeasurement] = Field(default_factory=dict)
    cumulative_session_usage: dict[str, int] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)


class CodeChunk(BaseModel):
    chunk_id: str
    file_path: str
    line_start: int
    line_end: int
    text: str
    token_estimate: int


class SourceSnippet(BaseModel):
    file_path: str
    line_start: int
    line_end: int
    text: str
    score: float | None = None
    source: str = "retrieval"


class ChatRequest(BaseModel):
    repo_id: str
    query: str = Field(min_length=1)
    session_id: str | None = None
    retrieval_method: str | None = None
    max_nodes: int | None = None
    max_anchors: int | None = None
    max_neighbors: int | None = None
    graphify_mode: str | None = None


class CompareRequest(ChatRequest):
    pass


class GitHubImportRequest(BaseModel):
    url: str = Field(min_length=1)


class ModelInfo(BaseModel):
    provider: str
    model: str
    configured: bool
    notes: str | None = None


class QueryRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    query_id: str
    repo_id: str
    session_id: str
    mode: Literal["standard", "graph_optimized"]
    query: str
    status: Literal["completed", "failed"]
    answer: str = ""
    error: str | None = None
    source_snippets: list[SourceSnippet] = Field(default_factory=list)
    selected_nodes: list[GraphNode] = Field(default_factory=list)
    selected_edges: list[GraphEdge] = Field(default_factory=list)
    token_usage: dict[str, TokenMeasurement] = Field(default_factory=dict)
    latency_ms: int = 0
    retrieval_strategy: str = "unknown"
    context: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class CompareResult(BaseModel):
    repo_id: str
    session_id: str
    query: str
    standard: QueryRecord
    graph_optimized: QueryRecord
    token_savings: dict[str, int | float | str]
    latency_delta_ms: int

