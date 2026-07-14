import { useNavigate } from "react-router-dom";
import { CheckCircle2, Download, FileText, X } from "lucide-react";
import { Dialog } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { exportDownloadUrl } from "@/lib/api";
import { formatDateTime } from "@/lib/utils";

export function ExportSuccessDialog({
  open,
  onClose,
  jobId,
  runId,
  fieldCount,
  pageCount,
  sourceFilename,
}: {
  open: boolean;
  onClose: () => void;
  jobId: string;
  runId: string | null;
  fieldCount: number;
  pageCount: number;
  sourceFilename: string;
}) {
  const navigate = useNavigate();
  const filledName = sourceFilename.replace(/\.pdf$/i, "") + "_filled.pdf";

  return (
    <Dialog open={open} onClose={onClose}>
      <button
        className="absolute right-4 top-4 rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
        onClick={onClose}
        aria-label="Close"
      >
        <X className="h-4 w-4" />
      </button>

      <div className="flex flex-col items-center text-center">
        <span className="flex h-14 w-14 items-center justify-center rounded-full bg-green-100">
          <CheckCircle2 className="h-8 w-8 text-green-600" />
        </span>
        <h2 className="mt-4 text-xl font-semibold text-slate-900">Your PDF is ready</h2>
        <p className="mt-1 text-sm text-slate-500">
          All {fieldCount} fields were stamped onto the original document. The output is flattened and ready to
          share.
        </p>
      </div>

      <div className="mt-5 flex items-center gap-3 rounded-xl border border-slate-200 px-4 py-3">
        <FileText className="h-6 w-6 shrink-0 text-red-500" />
        <div className="min-w-0 flex-1 text-left">
          <p className="truncate text-sm font-medium text-slate-800">{filledName}</p>
          <p className="text-xs text-slate-400">{pageCount} pages</p>
        </div>
      </div>

      <a href={exportDownloadUrl(jobId)} className="mt-4 block" download>
        <Button className="w-full" size="default">
          <Download className="h-4 w-4" />
          Download PDF
        </Button>
      </a>

      <div className="mt-4 flex items-center justify-center gap-4 text-sm">
        <button className="font-medium text-brand-600 hover:underline" onClick={onClose}>
          Back to editor
        </button>
        <span className="text-slate-300">|</span>
        <button className="font-medium text-brand-600 hover:underline" onClick={() => navigate("/")}>
          Go to documents
        </button>
      </div>

      <p className="mt-4 text-center text-xs text-slate-400">
        Exported {formatDateTime(new Date().toISOString())} &bull; Run {runId ?? "\u2014"}
      </p>
    </Dialog>
  );
}
