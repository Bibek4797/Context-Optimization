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
  uploadRepo(files: File[]) {
    if (files.length === 0) {
      throw new Error("Choose a zip file, source files, or a folder to upload.");
    }
    const form = new FormData();
    if (files.length === 1 && files[0].name.toLowerCase().endsWith(".zip")) {
      form.append("file", files[0]);
      return request<RepoMetadata>("/repo/upload", { method: "POST", body: form });
    }
    files.forEach((file) => {
      const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
      form.append("files", file, relativePath);
      form.append("paths", relativePath);
    });
    return request<RepoMetadata>("/repo/upload-files", { method: "POST", body: form });
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
