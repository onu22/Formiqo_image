import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { CheckCircle2, Loader2 } from "lucide-react";
import type { JobDetail } from "@/lib/types";
import { shortId } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { ProgressRing } from "@/components/ui/progress-ring";
import { api } from "@/lib/api";

type StepStatus = "done" | "active" | "waiting" | "failed";

interface Step {
  title: string;
  subtitle: string;
  status: StepStatus;
  progressPct?: number;
}

function buildSteps(job: JobDetail): Step[] {
  const grounding = job.stages.grounding;
  const convertDone = job.status !== "converting";
  const groundingDone = job.status === "ready" || job.status === "exported" || job.status === "failed";
  const groundingActive = job.status === "grounding";

  const groundedPages = grounding?.grounded_pages ?? 0;
  const totalPages = grounding?.total_pages ?? job.page_count;
  const groundingPct = totalPages > 0 ? Math.round((groundedPages / totalPages) * 100) : 0;

  return [
    {
      title: "Convert pages to images",
      subtitle: convertDone
        ? `${job.page_count || totalPages} pages rendered`
        : "Rendering pages...",
      status: convertDone ? "done" : "active",
    },
    {
      title: "Detecting form fields with AI",
      subtitle: groundingActive
        ? `Page ${Math.min(groundedPages + 1, Math.max(totalPages, 1))} of ${totalPages || "?"}`
        : groundingDone
          ? "Done"
          : "Waiting",
      status: groundingDone ? "done" : groundingActive ? "active" : "waiting",
      progressPct: groundingActive ? groundingPct : undefined,
    },
    {
      title: "Generate review layout",
      subtitle: groundingDone ? "Ready" : "Waiting",
      status: groundingDone ? "done" : "waiting",
    },
  ];
}

export function ProcessingView({ job, onCancelled }: { job: JobDetail; onCancelled: () => void }) {
  const navigate = useNavigate();
  const [confirmCancel, setConfirmCancel] = useState(false);
  const steps = buildSteps(job);
  const completedCount = steps.filter((s) => s.status === "done").length;

  async function cancelJob() {
    await api.deleteJob(job.job_id);
    setConfirmCancel(false);
    onCancelled();
    navigate("/");
  }

  return (
    <div className="flex justify-center px-6 py-12">
      <div className="w-full max-w-xl rounded-2xl bg-white p-10 shadow-sm">
        <div className="flex flex-col items-center">
          <ProgressRing value={completedCount} max={steps.length} label={`${completedCount}/${steps.length}`} />
          <h1 className="mt-6 text-xl font-semibold text-slate-900">Preparing your form...</h1>
          <p className="mt-1 text-center text-sm text-slate-500">
            This usually takes 1-2 minutes per page. You can leave this page and come back.
          </p>
        </div>

        <ol className="mt-8 space-y-0">
          {steps.map((step, idx) => (
            <li key={step.title} className="relative flex gap-4 pb-8 last:pb-0">
              {idx < steps.length - 1 && (
                <span
                  className={`absolute left-[15px] top-8 h-full w-px ${
                    step.status === "done" ? "bg-green-300" : "bg-slate-200"
                  }`}
                />
              )}
              <span className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full">
                {step.status === "done" && (
                  <CheckCircle2 className="h-8 w-8 text-green-500" fill="currentColor" />
                )}
                {step.status === "active" && (
                  <span className="flex h-8 w-8 items-center justify-center rounded-full bg-brand-600">
                    <Loader2 className="h-4 w-4 animate-spin text-white" />
                  </span>
                )}
                {step.status === "waiting" && <span className="h-6 w-6 rounded-full border-2 border-slate-300" />}
              </span>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <p
                    className={`text-sm font-semibold ${
                      step.status === "waiting" ? "text-slate-400" : "text-slate-900"
                    }`}
                  >
                    {step.title}
                  </p>
                  {step.status === "done" && idx === 0 && <span className="text-xs text-slate-400">Done</span>}
                  {step.status === "active" && step.progressPct !== undefined && (
                    <span className="text-xs font-medium text-brand-600">{step.progressPct}%</span>
                  )}
                </div>
                <p className={`text-xs ${step.status === "waiting" ? "text-slate-400" : "text-slate-500"}`}>
                  {step.subtitle}
                </p>
                {step.status === "active" && step.progressPct !== undefined && (
                  <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
                    <div
                      className="h-full rounded-full bg-brand-600 transition-all"
                      style={{ width: `${step.progressPct}%` }}
                    />
                  </div>
                )}
              </div>
            </li>
          ))}
        </ol>

        <div className="mt-4 flex items-center justify-between border-t border-slate-200 pt-5">
          <p className="text-xs text-slate-400">
            {job.page_count > 0 ? `Detected ${job.page_count} pages` : "Detecting pages"} &bull; Job{" "}
            {shortId(job.job_id)}
          </p>
          <Button variant="outline" onClick={() => setConfirmCancel(true)}>
            Cancel job
          </Button>
        </div>
      </div>

      <Dialog open={confirmCancel} onClose={() => setConfirmCancel(false)}>
        <h2 className="text-lg font-semibold text-slate-900">Cancel this job?</h2>
        <p className="mt-2 text-sm text-slate-600">
          This stops processing and deletes the job. This action cannot be undone.
        </p>
        <div className="mt-6 flex justify-end gap-3">
          <Button variant="outline" onClick={() => setConfirmCancel(false)}>
            Keep processing
          </Button>
          <Button variant="danger" onClick={cancelJob}>
            Cancel job
          </Button>
        </div>
      </Dialog>
    </div>
  );
}
