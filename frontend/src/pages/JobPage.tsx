import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { api } from "@/lib/api";
import type { JobDetail } from "@/lib/types";
import { FormiqoLogo } from "@/components/layout/logo";
import { StatusBadge } from "@/components/job/status-badge";
import { ProcessingView } from "@/components/job/processing-view";
import { FailedView } from "@/components/job/failed-view";
import { EditorView } from "@/pages/EditorView";

export function JobPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const forceEditor = searchParams.get("view") === "editor";

  const [job, setJob] = useState<JobDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const load = useCallback(async () => {
    if (!jobId) return;
    try {
      const detail = await api.getJob(jobId);
      setJob(detail);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load job");
    }
  }, [jobId]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!job) return;
    const isPolling = job.status === "converting" || job.status === "grounding";
    if (isPolling) {
      pollRef.current = window.setInterval(load, 2000);
    }
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
    // Only re-arm the poller when status flips (e.g. converting -> grounding -> ready).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.status, load]);

  if (error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-3 text-sm text-slate-500">
        <p>Could not load this job.</p>
        <p className="text-red-600">{error}</p>
        <button className="text-brand-600 hover:underline" onClick={() => navigate("/")}>
          Back to documents
        </button>
      </div>
    );
  }

  if (!job) {
    return <div className="flex h-screen items-center justify-center text-sm text-slate-400">Loading...</div>;
  }

  if (job.status === "ready" || job.status === "exported" || (job.status === "failed" && forceEditor)) {
    return <EditorView jobId={job.job_id} sourceFilename={job.source_filename ?? "document.pdf"} />;
  }

  return (
    <div className="min-h-full">
      <header className="flex items-center gap-4 border-b border-slate-200 bg-white px-6 py-4">
        <FormiqoLogo />
        <span className="h-5 w-px bg-slate-200" />
        <button
          className="flex items-center gap-1.5 text-sm font-medium text-brand-600 hover:underline"
          onClick={() => navigate("/")}
        >
          <ArrowLeft className="h-4 w-4" />
          Back to documents
        </button>
        <span className="h-5 w-px bg-slate-200" />
        <span className="text-sm font-medium text-slate-700">{job.source_filename}</span>
        <StatusBadge status={job.status} />
      </header>

      {job.status === "failed" ? (
        <FailedView job={job} onRefresh={load} />
      ) : (
        <ProcessingView job={job} onCancelled={() => navigate("/")} />
      )}
    </div>
  );
}

export default JobPage;
