from __future__ import annotations

from pathlib import Path

from app.services.analysis_pipeline import AnalysisPipeline
from app.services.codegraph_service import CodeGraphService
from app.services.graphify_service import GraphifyService
from app.services.storage import LocalStorage
from app.services.token_service import TokenService
from app.services.tree_sitter_service import TreeSitterService


def test_pipeline_analyzes_python_repo(tmp_path: Path) -> None:
    source = tmp_path / "sample"
    source.mkdir()
    (source / "app.py").write_text(
        "\n".join(
            [
                "import math",
                "",
                "class Calculator:",
                "    def square(self, value: int) -> int:",
                "        return multiply(value, value)",
                "",
                "def multiply(left: int, right: int) -> int:",
                "    return left * right",
                "",
                "def hypotenuse(a: float, b: float) -> float:",
                "    return math.sqrt(multiply(a, a) + multiply(b, b))",
            ]
        ),
        encoding="utf-8",
    )

    storage = LocalStorage(tmp_path / "data")
    token_service = TokenService()
    pipeline = AnalysisPipeline(
        storage=storage,
        tree_sitter_service=TreeSitterService(),
        codegraph_service=CodeGraphService(),
        graphify_service=GraphifyService(storage=storage, cli_name="__missing_graphify__"),
        token_service=token_service,
    )

    metadata = pipeline.analyze_existing("sample", source, "test")
    assert metadata.status in {"completed", "partial"}
    assert metadata.stats.python_files == 1

    files = storage.load_files(metadata.repo_id)
    assert files[0].parse_status == "parsed"

    tree = storage.load_tree_sitter(metadata.repo_id, "app.py")
    assert tree is not None
    assert tree.root is not None
    assert tree.root.type == "module"

    codegraph = storage.load_codegraph(metadata.repo_id)
    assert codegraph is not None
    labels = {node.label for node in codegraph.nodes}
    assert {"Calculator", "multiply", "hypotenuse"}.issubset(labels)
    assert any(edge.edge_type == "calls" for edge in codegraph.edges)

    graphify = storage.load_graphify(metadata.repo_id)
    assert graphify is not None
    assert graphify.source in {"graphify", "graphify-fallback"}
    assert isinstance(graphify.warnings, list)

    summary = storage.load_token_summary(metadata.repo_id)
    assert summary is not None
    assert len(summary.stages) > 0

