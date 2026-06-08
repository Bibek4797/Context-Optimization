export type RepoStatus = "pending" | "running" | "completed" | "partial" | "failed";
export type CountType = "exact" | "estimated";

export interface RepoStats {
  total_files: number;
  python_files: number;
  total_lines: number;
  python_lines: number;
}

export interface RepoMetadata {
  repo_id: string;
  name: string;
  origin: string;
  status: RepoStatus;
  stats: RepoStats;
  warnings: string[];
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface RepoFile {
  path: string;
  language: "python";
  size_bytes: number;
  line_count: number;
  parse_status: "pending" | "parsed" | "failed";
  parse_error?: string | null;
}

export interface TreeNode {
  type: string;
  named: boolean;
  start_point: [number, number];
  end_point: [number, number];
  start_byte: number;
  end_byte: number;
  text_preview?: string | null;
  children: TreeNode[];
}

export interface TreeSitterDocument {
  repo_id: string;
  file_path: string;
  language: "python";
  root?: TreeNode | null;
  source: string;
  warnings: string[];
  parse_error?: string | null;
  generated_at: string;
}

export interface GraphNode {
  node_id: string;
  node_type: string;
  label: string;
  file_path?: string | null;
  line_start?: number | null;
  line_end?: number | null;
  source_snippet?: string | null;
  metadata: Record<string, unknown>;
}

export interface GraphEdge {
  edge_id: string;
  edge_type: string;
  source_node: string;
  target_node: string;
  score?: number | null;
  metadata: Record<string, unknown>;
}

export interface GraphDocument {
  repo_id: string;
  source: "codegraph" | "graphify" | "graphify-fallback";
  nodes: GraphNode[];
  edges: GraphEdge[];
  raw_output_path?: string | null;
  warnings: string[];
  generated_at: string;
}

export interface TokenMeasurement {
  stage: string;
  tokens: number;
  count_type: CountType;
  provider?: string | null;
  model?: string | null;
  notes?: string | null;
}

export interface TokenSummary {
  repo_id: string;
  stages: Record<string, TokenMeasurement>;
  cumulative_session_usage: Record<string, number>;
  updated_at: string;
}

export interface SourceSnippet {
  file_path: string;
  line_start: number;
  line_end: number;
  text: string;
  score?: number | null;
  source: string;
}

export interface QueryRecord {
  query_id: string;
  repo_id: string;
  session_id: string;
  mode: "standard" | "graph_optimized";
  query: string;
  status: "completed" | "failed";
  answer: string;
  error?: string | null;
  source_snippets: SourceSnippet[];
  selected_nodes: GraphNode[];
  selected_edges: GraphEdge[];
  token_usage: Record<string, TokenMeasurement>;
  latency_ms: number;
  retrieval_strategy?: string;
  context?: string;
  created_at: string;
}

export interface CompareResult {
  repo_id: string;
  session_id: string;
  query: string;
  standard: QueryRecord;
  graph_optimized: QueryRecord;
  token_savings: Record<string, number | string>;
  latency_delta_ms: number;
}

export interface LogEntry {
  timestamp: string;
  stage: string;
  level: string;
  message: string;
  metadata: Record<string, unknown>;
}

