import { useEffect, useMemo, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Download, FileText, Pencil, Search, Trash2, Upload } from "lucide-react";
import { api, exportDownloadUrl } from "@/lib/api";
import type { JobListItem, JobStatus } from "@/lib/types";
import { formatDate, shortId } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FormiqoLogo } from "@/components/layout/logo";
import { StatusBadge } from "@/components/job/status-badge";
import { Dialog } from "@/components/ui/dialog";

const STATUS_FILTERS: { value: "all" | JobStatus; label: string }[] = [
  { value: "all", label: "All statuses" },
  { value: "ready", label: "Ready for review" },
  { value: "grounding", label: "Grounding fields" },
  { value: "converting", label: "Converting" },
  { value: "exported", label: "Exported" },
  { value: "failed", label: "Failed" },
];

export function JobsListPage() {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState<JobListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | JobStatus>("all");
  const [pendingDelete, setPendingDelete] = useState<JobListItem | null>(null);

  const loadJobs = useCallback(async () => {
    try {
      const res = await api.listJobs();
      setJobs(res.jobs);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load documents");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadJobs();
    const interval = setInterval(loadJobs, 5000);
    return () => clearInterval(interval);
  }, [loadJobs]);

  const filtered = useMemo(() => {
    return jobs.filter((job) => {
      if (statusFilter !== "all" && job.status !== statusFilter) return false;
      if (search.trim() && !job.source_filename.toLowerCase().includes(search.trim().toLowerCase())) {
        return false;
      }
      return true;
    });
  }, [jobs, search, statusFilter]);

  async function handleDelete(job: JobListItem) {
    await api.deleteJob(job.job_id);
    setPendingDelete(null);
    loadJobs();
  }

  return (
    <div className="min-h-full">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-4">
        <FormiqoLogo />
        <Button onClick={() => navigate("/upload")}>
          <Upload className="h-4 w-4" />
          Upload PDF
        </Button>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold text-slate-900">Documents</h1>
            <p className="mt-1 text-sm text-slate-500">Review and export your filled PDF forms</p>
          </div>
          <div className="flex gap-3">
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <Input
                placeholder="Search documents..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-64 pl-9"
              />
            </div>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as "all" | JobStatus)}
              className="h-9 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-700 outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20"
            >
              {STATUS_FILTERS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
          {error && (
            <div className="border-b border-red-200 bg-red-50 px-6 py-3 text-sm text-red-700">{error}</div>
          )}
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-xs font-medium uppercase tracking-wide text-slate-500">
                <th className="px-6 py-3">Document</th>
                <th className="px-6 py-3">Pages</th>
                <th className="px-6 py-3">Status</th>
                <th className="px-6 py-3">Created</th>
                <th className="px-6 py-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {!loading && filtered.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-sm text-slate-400">
                    No documents yet. Upload a PDF to get started.
                  </td>
                </tr>
              )}
              {filtered.map((job) => (
                <tr key={job.job_id} className="border-b border-slate-100 last:border-b-0 hover:bg-slate-50/60">
                  <td className="px-6 py-3.5">
                    <div className="flex items-center gap-3">
                      <FileText className="h-6 w-6 shrink-0 text-red-500" />
                      <span className="font-medium text-slate-800">{job.source_filename}</span>
                    </div>
                  </td>
                  <td className="px-6 py-3.5 text-slate-600">{job.page_count}</td>
                  <td className="px-6 py-3.5">
                    {job.status === "failed" ? (
                      <div className="flex items-center gap-3">
                        <StatusBadge status={job.status} />
                        <button
                          className="text-sm font-medium text-brand-600 hover:underline"
                          onClick={() => navigate(`/jobs/${job.job_id}`)}
                        >
                          View error
                        </button>
                      </div>
                    ) : (
                      <StatusBadge status={job.status} />
                    )}
                  </td>
                  <td className="px-6 py-3.5 text-slate-500">{formatDate(job.created_at)}</td>
                  <td className="px-6 py-3.5">
                    <div className="flex items-center gap-2">
                      {job.status !== "failed" && (
                        <Button
                          variant="outlineBrand"
                          size="sm"
                          disabled={job.status === "converting" || job.status === "grounding"}
                          onClick={() => navigate(`/jobs/${job.job_id}`)}
                        >
                          <Pencil className="h-3.5 w-3.5" />
                          Open editor
                        </Button>
                      )}
                      {(job.status === "ready" || job.status === "exported") && (
                        <a href={exportDownloadUrl(job.job_id)} title="Download PDF">
                          <Button variant="outline" size="icon">
                            <Download className="h-4 w-4" />
                          </Button>
                        </a>
                      )}
                      <Button
                        variant="outline"
                        size="icon"
                        className="border-red-200 text-red-600 hover:bg-red-50"
                        onClick={() => setPendingDelete(job)}
                        title="Delete"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex items-center justify-between border-t border-slate-200 px-6 py-3 text-sm text-slate-500">
            <span>
              Showing {filtered.length} of {jobs.length} documents
            </span>
            <div className="flex items-center gap-1">
              <Button variant="outline" size="iconSm" disabled>
                <span aria-hidden>&lsaquo;</span>
              </Button>
              <span className="flex h-7 w-7 items-center justify-center rounded-md bg-brand-600 text-xs font-medium text-white">
                1
              </span>
              <Button variant="outline" size="iconSm" disabled>
                <span aria-hidden>&rsaquo;</span>
              </Button>
            </div>
          </div>
        </div>
      </main>

      <Dialog open={pendingDelete !== null} onClose={() => setPendingDelete(null)}>
        {pendingDelete && (
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Delete document?</h2>
            <p className="mt-2 text-sm text-slate-600">
              This will permanently remove{" "}
              <span className="font-medium text-slate-800">{pendingDelete.source_filename}</span> (job{" "}
              {shortId(pendingDelete.job_id)}) and all of its artifacts.
            </p>
            <div className="mt-6 flex justify-end gap-3">
              <Button variant="outline" onClick={() => setPendingDelete(null)}>
                Cancel
              </Button>
              <Button variant="danger" onClick={() => handleDelete(pendingDelete)}>
                Delete
              </Button>
            </div>
          </div>
        )}
      </Dialog>
    </div>
  );
}

export default JobsListPage;
