import type {
  CompareResult,
  GraphDocument,
  LogEntry,
  QueryRecord,
  RepoFile,
  RepoMetadata,
  TokenSummary,
  TreeSitterDocument,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: init?.body instanceof FormData ? init.headers : { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      message = payload.detail ?? message;
    } catch {
      // Keep status text when response is not JSON.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export const api = {
  uploadRepo(file: File) {
    const form = new FormData();
    form.append("file", file);
    return request<RepoMetadata>("/repo/upload", { method: "POST", body: form });
  },
  importGithub(url: string) {
    return request<RepoMetadata>("/repo/import-github", { method: "POST", body: JSON.stringify({ url }) });
  },
  repoStatus(repoId: string) {
    return request<RepoMetadata>(`/repo/${repoId}/status`);
  },
  files(repoId: string) {
    return request<{ repo_id: string; files: RepoFile[] }>(`/repo/${repoId}/files`);
  },
  treeSitter(repoId: string, filePath: string) {
    return request<TreeSitterDocument>(`/repo/${repoId}/tree-sitter?file=${encodeURIComponent(filePath)}`);
  },
  codegraph(repoId: string) {
    return request<GraphDocument>(`/repo/${repoId}/codegraph`);
  },
  graphify(repoId: string) {
    return request<GraphDocument>(`/repo/${repoId}/graphify`);
  },
  tokenSummary(repoId: string) {
    return request<TokenSummary>(`/repo/${repoId}/token-summary`);
  },
  logs(repoId: string) {
    return request<{ repo_id: string; logs: LogEntry[] }>(`/repo/${repoId}/logs`);
  },
  standard(repoId: string, query: string, sessionId?: string) {
    return request<QueryRecord>("/chat/standard", {
      method: "POST",
      body: JSON.stringify({ repo_id: repoId, query, session_id: sessionId }),
    });
  },
  graphOptimized(repoId: string, query: string, sessionId?: string) {
    return request<QueryRecord>("/chat/graph-optimized", {
      method: "POST",
      body: JSON.stringify({ repo_id: repoId, query, session_id: sessionId }),
    });
  },
  compare(repoId: string, query: string, sessionId?: string) {
    return request<CompareResult>("/chat/compare", {
      method: "POST",
      body: JSON.stringify({ repo_id: repoId, query, session_id: sessionId }),
    });
  },
};

