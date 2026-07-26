from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    total_tokens: int = 0


class RepoMetadata(BaseModel):
    repo_id: str
    name: str
    origin: str
    status: RepoStatus = RepoStatus.pending
    stats: RepoStats = Field(default_factory=RepoStats)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class RepoFile(BaseModel):
    path: str
    language: str
    size_bytes: int
    lines: int
    parse_status: str = "pending"
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
    source_node: str
    target_node: str
    edge_type: str
    score: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphDocument(BaseModel):
    repo_id: str
    source: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)


class TokenMeasurement(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True, extra="allow")
    stage: str = ""
    tokens: int = 0
    count_type: Any = "estimated"
    provider: str | None = None
    model: str | None = None
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _validate_measurement_before(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return data
        if hasattr(data, "tokens"):
            ct = getattr(data, "count_type", "estimated")
            ct_val = getattr(ct, "value", str(ct))
            return {
                "stage": str(getattr(data, "stage", "")),
                "tokens": int(getattr(data, "tokens", 0)),
                "count_type": ct_val,
                "provider": getattr(data, "provider", None),
                "model": getattr(data, "model", None),
                "notes": getattr(data, "notes", None),
            }
        return data


class TokenSummary(BaseModel):
    repo_id: str
    stages: dict[str, Any] = Field(default_factory=dict)
    cumulative_session_usage: dict[str, int] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def _validate_summary_before(cls, data: Any) -> Any:
        if isinstance(data, dict):
            stages = data.get("stages")
            if isinstance(stages, dict):
                clean_stages = {}
                for k, v in stages.items():
                    if isinstance(v, dict):
                        clean_stages[k] = v
                    elif hasattr(v, "tokens"):
                        ct = getattr(v, "count_type", "exact")
                        clean_stages[k] = {
                            "stage": str(getattr(v, "stage", str(k))),
                            "tokens": int(getattr(v, "tokens", 0)),
                            "count_type": getattr(ct, "value", str(ct)),
                            "provider": getattr(v, "provider", None),
                            "model": getattr(v, "model", None),
                            "notes": getattr(v, "notes", None),
                        }
                    else:
                        clean_stages[k] = {"stage": str(k), "tokens": 0, "count_type": "exact"}
                data["stages"] = clean_stages
        return data


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
    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True, extra="allow")

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
    token_usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    retrieval_strategy: str = "unknown"
    context: str = ""
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def _validate_query_record_before(cls, data: Any) -> Any:
        if isinstance(data, dict):
            tu = data.get("token_usage")
            if isinstance(tu, dict):
                clean_tu = {}
                for k, v in tu.items():
                    if isinstance(v, dict):
                        clean_tu[k] = v
                    elif hasattr(v, "tokens"):
                        ct = getattr(v, "count_type", "exact")
                        ct_val = getattr(ct, "value", str(ct))
                        clean_tu[k] = {
                            "stage": str(getattr(v, "stage", str(k))),
                            "tokens": int(getattr(v, "tokens", 0)),
                            "count_type": ct_val,
                            "provider": getattr(v, "provider", None),
                            "model": getattr(v, "model", None),
                            "notes": getattr(v, "notes", None),
                        }
                    else:
                        clean_tu[k] = {"stage": str(k), "tokens": 0, "count_type": "exact"}
                data["token_usage"] = clean_tu
        return data


class CompareResult(BaseModel):
    repo_id: str
    session_id: str
    query: str
    standard: QueryRecord
    graph_optimized: QueryRecord
    token_savings: dict[str, int | float | str]
    latency_delta_ms: int

