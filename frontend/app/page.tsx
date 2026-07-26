"use client";

import {
  AlertTriangle,
  BarChart3,
  Bot,
  Boxes,
  Code2,
  GitBranch,
  GitCompare,
  Github,
  Network,
  RefreshCw,
  Send,
  Upload,
} from "lucide-react";
import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { GraphView } from "@/components/GraphView";
import { TokenTable } from "@/components/TokenTable";
import { TreeExplorer } from "@/components/TreeExplorer";
import { api } from "@/lib/api";
import type {
  CompareResult,
  GraphDocument,
  LogEntry,
  QueryRecord,
  RepoFile,
  RepoMetadata,
  TokenMeasurement,
  TokenSummary,
  TreeSitterDocument,
} from "@/lib/types";

type TabId =
  | "upload"
  | "analysis"
  | "tree"
  | "codegraph"
  | "graphify"
  | "tokens"
  | "standard"
  | "optimized"
  | "compare";

const tabs: { id: TabId; label: string; icon: ReactNode }[] = [
  { id: "upload", label: "Upload/Import", icon: <Upload size={16} /> },
  { id: "analysis", label: "Repo Analysis", icon: <Code2 size={16} /> },
  { id: "tree", label: "Tree-sitter", icon: <Boxes size={16} /> },
  { id: "codegraph", label: "CodeGraph", icon: <Network size={16} /> },
  { id: "graphify", label: "Graphify", icon: <GitBranch size={16} /> },
  { id: "tokens", label: "Token Analytics", icon: <BarChart3 size={16} /> },
  { id: "standard", label: "Standard QA", icon: <Bot size={16} /> },
  { id: "optimized", label: "Graph QA", icon: <Bot size={16} /> },
  { id: "compare", label: "Compare", icon: <GitCompare size={16} /> },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabId>("upload");
  const [repo, setRepo] = useState<RepoMetadata | null>(null);
  const [files, setFiles] = useState<RepoFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>("");
  const [tree, setTree] = useState<TreeSitterDocument | null>(null);
  const [codegraph, setCodegraph] = useState<GraphDocument | null>(null);
  const [graphify, setGraphify] = useState<GraphDocument | null>(null);
  const [tokens, setTokens] = useState<TokenSummary | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [githubUrl, setGithubUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [sessionId] = useState(() => crypto.randomUUID());

  const [standardQuery, setStandardQuery] = useState("");
  const [optimizedQuery, setOptimizedQuery] = useState("");
  const [compareQuery, setCompareQuery] = useState("");
  const [standardResult, setStandardResult] = useState<QueryRecord | null>(null);
  const [optimizedResult, setOptimizedResult] = useState<QueryRecord | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null);

  const totalSessionTokens = useMemo(() => {
    if (!tokens) return 0;
    return Object.values(tokens.cumulative_session_usage).reduce((sum, value) => sum + value, 0);
  }, [tokens]);

  async function hydrateRepo(repoId: string, nextTab: TabId = "analysis") {
    setMessage(null);
    const [status, filePayload, codegraphPayload, graphifyPayload, tokenPayload, logPayload] = await Promise.all([
      api.repoStatus(repoId),
      api.files(repoId),
      api.codegraph(repoId).catch(() => null),
      api.graphify(repoId).catch(() => null),
      api.tokenSummary(repoId).catch(() => null),
      api.logs(repoId).catch(() => ({ logs: [] })),
    ]);
    setRepo(status);
    setFiles(filePayload.files);
    setCodegraph(codegraphPayload);
    setGraphify(graphifyPayload);
    setTokens(tokenPayload);
    setLogs(logPayload.logs);
    const firstFile = filePayload.files[0]?.path ?? "";
    setSelectedFile((current) => current || firstFile);
    if (firstFile) {
      const treePayload = await api.treeSitter(repoId, firstFile).catch(() => null);
      setTree(treePayload);
    }
    setActiveTab(nextTab);
  }

  async function handleUpload(file: File | null) {
    if (!file) return;
    setBusy(true);
    setMessage("Uploading and analyzing repository...");
    try {
      const metadata = await api.uploadRepo(file);
      await hydrateRepo(metadata.repo_id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleImport() {
    if (!githubUrl.trim()) return;
    setBusy(true);
    setMessage("Cloning and analyzing repository...");
    try {
      const metadata = await api.importGithub(githubUrl.trim());
      await hydrateRepo(metadata.repo_id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Import failed.");
    } finally {
      setBusy(false);
    }
  }

  async function loadTree(filePath: string) {
    if (!repo) return;
    setSelectedFile(filePath);
    setTree(await api.treeSitter(repo.repo_id, filePath));
  }

  async function refresh() {
    if (!repo) return;
    setBusy(true);
    try {
      await hydrateRepo(repo.repo_id, activeTab);
    } finally {
      setBusy(false);
    }
  }

  async function askStandard() {
    if (!repo || !standardQuery.trim()) return;
    setBusy(true);
    try {
      const result = await api.standard(repo.repo_id, standardQuery.trim(), sessionId);
      setStandardResult(result);
      setTokens(await api.tokenSummary(repo.repo_id));
      setLogs((await api.logs(repo.repo_id)).logs);
    } finally {
      setBusy(false);
    }
  }

  async function askOptimized() {
    if (!repo || !optimizedQuery.trim()) return;
    setBusy(true);
    try {
      const result = await api.graphOptimized(repo.repo_id, optimizedQuery.trim(), sessionId);
      setOptimizedResult(result);
      setTokens(await api.tokenSummary(repo.repo_id));
      setLogs((await api.logs(repo.repo_id)).logs);
    } finally {
      setBusy(false);
    }
  }

  async function runCompare() {
    if (!repo || !compareQuery.trim()) return;
    setBusy(true);
    try {
      const result = await api.compare(repo.repo_id, compareQuery.trim(), sessionId);
      setCompareResult(result);
      setTokens(await api.tokenSummary(repo.repo_id));
      setLogs((await api.logs(repo.repo_id)).logs);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#eef1ed] text-ink">
      <div className="grid min-h-screen grid-cols-1 lg:grid-cols-[280px_1fr]">
        <aside className="border-r border-line bg-[#fbfcfa] p-4">
          <div className="mb-5">
            <p className="text-xs uppercase tracking-wide text-zinc-500">Context Engine</p>
            <h1 className="mt-1 text-xl font-semibold">Python Repo Optimizer</h1>
          </div>

          <div className="mb-5 border border-line bg-white p-3">
            <p className="text-xs uppercase tracking-wide text-zinc-500">Repository</p>
            <p className="mt-1 truncate font-medium">{repo?.name ?? "No repo loaded"}</p>
            <p className="mt-1 text-xs text-zinc-500">{repo?.status ?? "idle"}</p>
            {repo ? (
              <button
                type="button"
                onClick={refresh}
                className="mt-3 inline-flex items-center gap-2 border border-line px-2 py-1 text-xs hover:bg-panel"
              >
                <RefreshCw size={13} /> Refresh
              </button>
            ) : null}
          </div>

          <nav className="space-y-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`flex w-full items-center gap-2 border px-3 py-2 text-left text-sm ${
                  activeTab === tab.id ? "border-accent bg-[#e6f4f1] text-accent" : "border-transparent hover:border-line hover:bg-white"
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </nav>
        </aside>

        <section className="flex min-w-0 flex-col">
          <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line bg-white px-5 py-3">
            <div className="flex flex-wrap gap-2">
              <StatusPill label="Model" value="Gemini env" />
              <StatusPill label="Files" value={repo ? `${repo.stats.python_files} Python` : "0 Python"} />
              <StatusPill label="Session Tokens" value={totalSessionTokens.toLocaleString()} />
              <StatusPill label="Graphify" value={graphify?.source ?? "not loaded"} />
            </div>
            {busy ? <span className="text-sm text-accent">Working...</span> : null}
          </header>

          <div className="flex-1 overflow-auto p-5">
            {message ? (
              <div className="mb-4 flex items-center gap-2 border border-line bg-white p-3 text-sm">
                <AlertTriangle size={16} className="text-rust" />
                {message}
              </div>
            ) : null}
            {repo?.warnings?.length ? <Warnings warnings={repo.warnings} /> : null}

            {activeTab === "upload" ? (
              <UploadImportPanel
                githubUrl={githubUrl}
                setGithubUrl={setGithubUrl}
                onUpload={handleUpload}
                onImport={handleImport}
                busy={busy}
              />
            ) : null}

            {activeTab === "analysis" ? <RepoAnalysis repo={repo} files={files} logs={logs} /> : null}

            {activeTab === "tree" ? (
              <Panel title="Tree-sitter Explorer">
                <div className="mb-3 flex flex-wrap gap-2">
                  {files.map((file) => (
                    <button
                      key={file.path}
                      type="button"
                      onClick={() => loadTree(file.path)}
                      className={`border px-3 py-1 text-xs ${
                        selectedFile === file.path ? "border-accent bg-[#e6f4f1] text-accent" : "border-line bg-white"
                      }`}
                    >
                      {file.path}
                    </button>
                  ))}
                </div>
                <TreeExplorer document={tree} />
              </Panel>
            ) : null}

            {activeTab === "codegraph" ? <GraphPanel title="CodeGraph Explorer" graph={codegraph} /> : null}

            {activeTab === "graphify" ? <GraphPanel title="Graphify Explorer" graph={graphify} /> : null}

            {activeTab === "tokens" ? (
              <Panel title="Token Analytics">
                <TokenTable stages={tokens?.stages ?? {}} />
                <div className="mt-4 grid gap-3 md:grid-cols-3">
                  {Object.entries(tokens?.cumulative_session_usage ?? {}).map(([key, value]) => (
                    <Metric key={key} label={key} value={value.toLocaleString()} />
                  ))}
                </div>
              </Panel>
            ) : null}

            {activeTab === "standard" ? (
              <QaPanel
                title="Standard Repo QA"
                query={standardQuery}
                setQuery={setStandardQuery}
                onAsk={askStandard}
                result={standardResult}
                busy={busy}
                disabled={!repo}
              />
            ) : null}

            {activeTab === "optimized" ? (
              <QaPanel
                title="Graph-Optimized Repo QA"
                query={optimizedQuery}
                setQuery={setOptimizedQuery}
                onAsk={askOptimized}
                result={optimizedResult}
                busy={busy}
                disabled={!repo}
                showNodes
              />
            ) : null}

            {activeTab === "compare" ? (
              <ComparePanel
                query={compareQuery}
                setQuery={setCompareQuery}
                onCompare={runCompare}
                result={compareResult}
                busy={busy}
                disabled={!repo}
              />
            ) : null}
          </div>
        </section>
      </div>
    </main>
  );
}

function UploadImportPanel({
  githubUrl,
  setGithubUrl,
  onUpload,
  onImport,
  busy,
}: {
  githubUrl: string;
  setGithubUrl: (value: string) => void;
  onUpload: (file: File | null) => void;
  onImport: () => void;
  busy: boolean;
}) {
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Panel title="Upload Zip">
        <label className="flex min-h-52 cursor-pointer flex-col items-center justify-center border border-dashed border-line bg-white p-8 text-center hover:bg-panel">
          <Upload size={28} className="mb-3 text-accent" />
          <span className="font-medium">Choose zipped Python repo</span>
          <span className="mt-1 text-sm text-zinc-500">Ignored folders are skipped during extraction.</span>
          <input type="file" accept=".zip" className="hidden" onChange={(event) => onUpload(event.target.files?.[0] ?? null)} disabled={busy} />
        </label>
      </Panel>

      <Panel title="Import GitHub">
        <div className="flex min-h-52 flex-col justify-center gap-3 border border-line bg-white p-5">
          <label className="text-sm font-medium" htmlFor="github-url">
            GitHub repository URL
          </label>
          <div className="flex flex-col gap-2 md:flex-row">
            <input
              id="github-url"
              value={githubUrl}
              onChange={(event) => setGithubUrl(event.target.value)}
              placeholder="https://github.com/owner/repo"
              className="min-w-0 flex-1 border border-line bg-white px-3 py-2 outline-none focus:border-accent"
            />
            <button
              type="button"
              disabled={busy || !githubUrl.trim()}
              onClick={onImport}
              className="inline-flex items-center justify-center gap-2 border border-accent bg-accent px-4 py-2 text-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Github size={16} /> Import
            </button>
          </div>
        </div>
      </Panel>
    </div>
  );
}

function RepoAnalysis({ repo, files, logs }: { repo: RepoMetadata | null; files: RepoFile[]; logs: LogEntry[] }) {
  if (!repo) return <EmptyState text="Load a Python repository to inspect analysis artifacts." />;
  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-4">
        <Metric label="Total Files" value={repo.stats.total_files.toLocaleString()} />
        <Metric label="Python Files" value={repo.stats.python_files.toLocaleString()} />
        <Metric label="Total Lines" value={repo.stats.total_lines.toLocaleString()} />
        <Metric label="Python Lines" value={repo.stats.python_lines.toLocaleString()} />
      </div>
      <Panel title="Files">
        <div className="max-h-[360px] overflow-auto border border-line bg-white scrollbar-thin">
          <table className="w-full text-left text-sm">
            <thead className="bg-panel text-xs uppercase tracking-wide text-zinc-500">
              <tr>
                <th className="px-3 py-2">Path</th>
                <th className="px-3 py-2">Lines</th>
                <th className="px-3 py-2">Parse</th>
              </tr>
            </thead>
            <tbody>
              {files.map((file) => (
                <tr key={file.path} className="border-t border-line">
                  <td className="px-3 py-2 font-mono text-xs">{file.path}</td>
                  <td className="px-3 py-2 tabular-nums">{file.line_count}</td>
                  <td className="px-3 py-2">{file.parse_status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <DebugPanel logs={logs} />
    </div>
  );
}

function GraphPanel({ title, graph }: { title: string; graph: GraphDocument | null }) {
  return (
    <Panel title={title}>
      {graph?.warnings?.length ? <Warnings warnings={graph.warnings} /> : null}
      <div className="mb-3 grid gap-3 md:grid-cols-4">
        <Metric label="Source" value={graph?.source ?? "none"} />
        <Metric label="Nodes" value={(graph?.nodes.length ?? 0).toLocaleString()} />
        <Metric label="Edges" value={(graph?.edges.length ?? 0).toLocaleString()} />
        <Metric label="Raw Output" value={graph?.raw_output_path ? "saved" : "none"} />
      </div>
      <GraphView graph={graph} />
      {graph ? (
        <div className="mt-4 grid gap-4 xl:grid-cols-2">
          <GraphList title="Nodes" items={graph.nodes.slice(0, 80).map((node) => `${node.node_type} | ${node.label} | ${node.file_path ?? "external"}`)} />
          <GraphList title="Edges" items={graph.edges.slice(0, 80).map((edge) => `${edge.edge_type} | ${edge.source_node} -> ${edge.target_node}`)} />
        </div>
      ) : null}
    </Panel>
  );
}

function QaPanel({
  title,
  query,
  setQuery,
  onAsk,
  result,
  busy,
  disabled,
  showNodes = false,
}: {
  title: string;
  query: string;
  setQuery: (value: string) => void;
  onAsk: () => void;
  result: QueryRecord | null;
  busy: boolean;
  disabled: boolean;
  showNodes?: boolean;
}) {
  return (
    <Panel title={title}>
      <div className="flex flex-col gap-2">
        <textarea
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          rows={4}
          placeholder="Ask about architecture, call paths, responsibilities, or implementation details."
          className="border border-line bg-white p-3 outline-none focus:border-accent"
        />
        <button
          type="button"
          onClick={onAsk}
          disabled={busy || disabled || !query.trim()}
          className="inline-flex w-fit items-center gap-2 border border-accent bg-accent px-4 py-2 text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Send size={16} /> Ask
        </button>
      </div>
      {result ? <QueryResult result={result} showNodes={showNodes} /> : null}
    </Panel>
  );
}

function ComparePanel({
  query,
  setQuery,
  onCompare,
  result,
  busy,
  disabled,
}: {
  query: string;
  setQuery: (value: string) => void;
  onCompare: () => void;
  result: CompareResult | null;
  busy: boolean;
  disabled: boolean;
}) {
  return (
    <Panel title="Baseline vs Optimized">
      <div className="flex flex-col gap-2">
        <textarea
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          rows={4}
          placeholder="Run the same query through standard and graph-optimized context."
          className="border border-line bg-white p-3 outline-none focus:border-accent"
        />
        <button
          type="button"
          onClick={onCompare}
          disabled={busy || disabled || !query.trim()}
          className="inline-flex w-fit items-center gap-2 border border-accent bg-accent px-4 py-2 text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          <GitCompare size={16} /> Compare
        </button>
      </div>
      {result ? (
        <div className="mt-5 space-y-4">
          <div className="grid gap-3 md:grid-cols-4">
            <Metric label="Baseline Context" value={`${result.token_savings.baseline_context_tokens ?? 0}`} />
            <Metric label="Optimized Context" value={`${result.token_savings.optimized_context_tokens ?? 0}`} />
            <Metric label="Saved Tokens" value={`${result.token_savings.saved_context_tokens ?? 0}`} />
            <Metric label="Saved Percent" value={`${result.token_savings.saved_percent ?? 0}%`} />
          </div>
          <div className="grid gap-4 xl:grid-cols-2">
            <QueryResult result={result.standard} compactTitle="Standard" />
            <QueryResult result={result.graph_optimized} compactTitle="Graph Optimized" showNodes />
          </div>
        </div>
      ) : null}
    </Panel>
  );
}

function QueryResult({ result, showNodes = false, compactTitle }: { result: QueryRecord; showNodes?: boolean; compactTitle?: string }) {
  return (
    <div className="mt-5 border border-line bg-white p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-semibold">{compactTitle ?? "Answer"}</h3>
        <span className={`border px-2 py-1 text-xs ${result.status === "completed" ? "border-accent text-accent" : "border-rust text-rust"}`}>
          {result.status} | {result.latency_ms} ms
        </span>
      </div>
      {result.error ? <p className="mb-3 border border-rust bg-[#fff7f3] p-3 text-sm text-rust">{result.error}</p> : null}
      <div className="whitespace-pre-wrap text-sm leading-6">{result.answer || "No answer generated."}</div>
      <div className="mt-4">
        <TokenTable stages={result.token_usage} />
      </div>
      {showNodes && result.selected_nodes.length ? (
        <GraphList title="Selected Nodes" items={result.selected_nodes.slice(0, 20).map((node) => `${node.node_type} | ${node.label} | ${node.file_path ?? "external"}`)} />
      ) : null}
      <GraphList
        title="Source Snippets"
        items={result.source_snippets.slice(0, 12).map((snippet) => `${snippet.file_path}:${snippet.line_start}-${snippet.line_end}\n${snippet.text}`)}
      />
    </div>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold">{title}</h2>
      </div>
      {children}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-line bg-white p-3">
      <p className="text-xs uppercase tracking-wide text-zinc-500">{label}</p>
      <p className="mt-1 truncate text-lg font-semibold">{value}</p>
    </div>
  );
}

function StatusPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-line bg-panel px-3 py-1 text-xs">
      <span className="text-zinc-500">{label}: </span>
      <span className="font-medium">{value}</span>
    </div>
  );
}

function GraphList({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="mt-4">
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      <div className="max-h-72 overflow-auto border border-line bg-[#fbfcfa] p-3 font-mono text-xs leading-5 scrollbar-thin">
        {items.length ? items.map((item, index) => <pre key={`${title}-${index}`} className="mb-2 whitespace-pre-wrap border-b border-line pb-2 last:mb-0 last:border-b-0">{item}</pre>) : <p className="font-sans text-sm text-zinc-500">None.</p>}
      </div>
    </div>
  );
}

function Warnings({ warnings }: { warnings: string[] }) {
  return (
    <div className="mb-4 space-y-2">
      {warnings.map((warning, index) => (
        <div key={`${warning}-${index}`} className="flex items-start gap-2 border border-rust bg-[#fff7f3] p-3 text-sm text-rust">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" />
          <span>{warning}</span>
        </div>
      ))}
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return <div className="border border-dashed border-line bg-white p-8 text-sm text-zinc-500">{text}</div>;
}

function DebugPanel({ logs }: { logs: LogEntry[] }) {
  return (
    <Panel title="Developer Debug Panel">
      <div className="max-h-80 overflow-auto border border-line bg-[#151718] p-3 font-mono text-xs leading-5 text-[#dbe5dd] scrollbar-thin">
        {logs.length ? (
          logs.map((log, index) => (
            <div key={`${log.timestamp}-${index}`}>
              <span className="text-[#89d6ca]">{log.timestamp}</span> <span className="text-[#f1bf98]">{log.level}</span>{" "}
              <span className="text-[#d8c4e2]">{log.stage}</span> {log.message}
            </div>
          ))
        ) : (
          <span>No logs yet.</span>
        )}
      </div>
    </Panel>
  );
}
