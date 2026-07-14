import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { CheckCircle2, RotateCw, XCircle } from "lucide-react";
import type { JobDetail, JobStageError } from "@/lib/types";
import { shortId } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { api } from "@/lib/api";

function formatError(err: JobStageError): string {
  if (typeof err.page_index === "number") {
    const detail = err.detail || err.error || "vision model returned invalid field data.";
    return `Grounding failed on page ${err.page_index + 1}: ${detail}`;
  }
  if (err.stage && err.message) {
    return err.message;
  }
  return JSON.stringify(err);
}

export function FailedView({ job, onRefresh }: { job: JobDetail; onRefresh: () => void }) {
  const navigate = useNavigate();
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const [retryError, setRetryError] = useState<string | null>(null);

  const grounding = job.stages.grounding;
  const totalPages = grounding?.total_pages ?? job.page_count;
  const groundedPages = grounding?.grounded_pages ?? 0;
  const failedCount = Math.max(totalPages - groundedPages, 0);

  async function retryFailedPages() {
    setRetrying(true);
    setRetryError(null);
    try {
      await api.retryGrounding(job.job_id);
      onRefresh();
    } catch (err) {
      setRetryError(err instanceof Error ? err.message : "Retry failed. Please try again.");
    } finally {
      setRetrying(false);
    }
  }

  async function deleteJob() {
    await api.deleteJob(job.job_id);
    navigate("/");
  }

  return (
    <div className="flex justify-center px-6 py-12">
      <div className="w-full max-w-xl rounded-2xl bg-white p-10 shadow-sm">
        <div className="flex flex-col items-center">
          <span className="flex h-16 w-16 items-center justify-center rounded-full bg-red-100">
            <XCircle className="h-9 w-9 text-red-500" />
          </span>
          <h1 className="mt-5 text-xl font-semibold text-slate-900">We couldn&apos;t process this form</h1>
          <p className="mt-2 max-w-sm text-center text-sm text-slate-500">
            Field detection failed on {failedCount} of {totalPages} pages. You can retry processing or review the
            pages that succeeded.
          </p>
        </div>

        {job.errors.length > 0 && (
          <div className="mt-6 space-y-1 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {job.errors.map((err, idx) => (
              <p key={idx}>{formatError(err)}</p>
            ))}
          </div>
        )}

        <ul className="mt-6 space-y-4">
          <li className="flex items-start gap-3">
            <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-green-500" fill="currentColor" />
            <div>
              <p className="text-sm font-semibold text-slate-900">Convert pages to images</p>
              <p className="text-xs text-slate-500">{job.page_count} pages rendered</p>
            </div>
          </li>
          <li className="flex items-start gap-3">
            <XCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-500" fill="currentColor" />
            <div>
              <p className="text-sm font-semibold text-slate-900">Detecting form fields with AI</p>
              <p className="text-xs text-slate-500">
                {groundedPages} of {totalPages} pages succeeded
              </p>
            </div>
          </li>
          <li className="flex items-start gap-3">
            <span className="mt-0.5 h-5 w-5 shrink-0 rounded-full border-2 border-slate-300" />
            <div>
              <p className="text-sm font-semibold text-slate-400">Generate review layout</p>
              <p className="text-xs text-slate-400">Skipped</p>
            </div>
          </li>
        </ul>

        {retryError && <p className="mt-4 text-sm text-red-600">{retryError}</p>}

        <div className="mt-8 flex flex-wrap items-center gap-3 border-t border-slate-200 pt-6">
          <Button onClick={retryFailedPages} disabled={retrying}>
            <RotateCw className={`h-4 w-4 ${retrying ? "animate-spin" : ""}`} />
            {retrying ? "Retrying..." : "Retry failed pages"}
          </Button>
          {groundedPages > 0 && (
            <Button variant="outline" onClick={() => navigate(`/jobs/${job.job_id}?view=editor`)}>
              Review {groundedPages} completed pages
            </Button>
          )}
          <Button variant="dangerGhost" className="ml-auto" onClick={() => setConfirmDelete(true)}>
            Delete job
          </Button>
        </div>

        <p className="mt-6 text-center text-xs text-slate-400">Job {shortId(job.job_id)}</p>
      </div>

      <Dialog open={confirmDelete} onClose={() => setConfirmDelete(false)}>
        <h2 className="text-lg font-semibold text-slate-900">Delete this job?</h2>
        <p className="mt-2 text-sm text-slate-600">This permanently removes the job and all of its artifacts.</p>
        <div className="mt-6 flex justify-end gap-3">
          <Button variant="outline" onClick={() => setConfirmDelete(false)}>
            Cancel
          </Button>
          <Button variant="danger" onClick={deleteJob}>
            Delete
          </Button>
        </div>
      </Dialog>
    </div>
  );
}
