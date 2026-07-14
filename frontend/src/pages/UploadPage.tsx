import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, FileText, UploadCloud, X } from "lucide-react";
import { ApiError, api } from "@/lib/api";
import { formatBytes } from "@/lib/utils";
import { FormiqoLogo } from "@/components/layout/logo";

const MAX_BYTES = 50 * 1024 * 1024;

type UploadState = "idle" | "uploading" | "error";

export function UploadPage() {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [state, setState] = useState<UploadState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const startUpload = useCallback(
    (candidate: File) => {
      setErrorMessage(null);

      if (!candidate.name.toLowerCase().endsWith(".pdf") || (candidate.type && candidate.type !== "application/pdf")) {
        setFile(candidate);
        setState("error");
        setErrorMessage("Only PDF files are supported.");
        return;
      }
      if (candidate.size > MAX_BYTES) {
        setFile(candidate);
        setState("error");
        setErrorMessage(`File is too large. Maximum upload size is ${formatBytes(MAX_BYTES)}.`);
        return;
      }

      setFile(candidate);
      setState("uploading");
      setProgress(0);

      // Simulated progress while the multipart request is in flight; the
      // backend does not stream progress events.
      const progressTimer = window.setInterval(() => {
        setProgress((p) => (p < 90 ? p + Math.random() * 18 : p));
      }, 220);

      api
        .createJob(candidate)
        .then((res) => {
          window.clearInterval(progressTimer);
          setProgress(100);
          window.setTimeout(() => navigate(`/jobs/${res.job_id}`), 250);
        })
        .catch((err) => {
          window.clearInterval(progressTimer);
          setState("error");
          if (err instanceof ApiError) {
            if (err.code === "xfa_not_supported") {
              setErrorMessage("XFA forms are not supported. Please flatten the PDF before uploading. Scanned and flat PDFs work great.");
            } else {
              setErrorMessage(err.message);
            }
          } else {
            setErrorMessage("Upload failed. Please try again.");
          }
        });
    },
    [navigate],
  );

  function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragActive(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) startUpload(dropped);
  }

  function onBrowse(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = e.target.files?.[0];
    if (picked) startUpload(picked);
  }

  function cancelUpload() {
    setFile(null);
    setState("idle");
    setProgress(0);
    setErrorMessage(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  const isXfaError = errorMessage?.toLowerCase().includes("xfa");

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
      </header>

      <main className="flex justify-center px-6 py-12">
        <div className="w-full max-w-xl rounded-2xl bg-white p-8 shadow-sm">
          <h1 className="text-center text-2xl font-semibold text-slate-900">Upload a PDF form</h1>
          <p className="mt-2 text-center text-sm text-slate-500">
            We&apos;ll detect the fillable fields automatically so you can review and edit them.
          </p>

          <div
            className={`mt-6 flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-14 text-center transition-colors ${
              dragActive ? "border-brand-500 bg-brand-50" : "border-brand-300 bg-brand-50/40"
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
          >
            <UploadCloud className="h-12 w-12 text-brand-400" strokeWidth={1.5} />
            <p className="mt-4 text-base font-semibold text-slate-800">Drop your PDF here</p>
            <button
              className="mt-1 text-sm font-medium text-brand-600 hover:underline"
              onClick={() => inputRef.current?.click()}
            >
              or click to browse files
            </button>
            <p className="mt-3 text-xs text-slate-400">PDF only &bull; up to 50 MB &bull; scanned forms supported</p>
            <input ref={inputRef} type="file" accept="application/pdf" className="hidden" onChange={onBrowse} />
          </div>

          {file && (
            <div className="mt-5 flex items-center gap-3 rounded-xl border border-slate-200 px-4 py-3">
              <FileText className="h-6 w-6 shrink-0 text-red-500" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-slate-800">{file.name}</p>
                <p className="text-xs text-slate-400">{formatBytes(file.size)}</p>
              </div>
              {state === "uploading" && (
                <div className="flex w-40 items-center gap-2">
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-200">
                    <div
                      className="h-full rounded-full bg-brand-600 transition-all"
                      style={{ width: `${Math.min(progress, 100)}%` }}
                    />
                  </div>
                  <span className="w-16 shrink-0 text-right text-xs font-medium text-brand-600">
                    Uploading... {Math.round(Math.min(progress, 100))}%
                  </span>
                </div>
              )}
              <button
                className="ml-1 rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
                onClick={cancelUpload}
                aria-label="Cancel upload"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {state === "error" && errorMessage && (
            <div className="mt-5 flex items-start gap-3 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              <span className="mt-0.5 text-amber-500">&#9888;</span>
              <div>
                <p className="font-semibold">{isXfaError ? "XFA forms are not supported." : "Upload failed"}</p>
                <p className="mt-0.5 text-amber-700/90">{errorMessage}</p>
              </div>
            </div>
          )}

          <p className="mt-6 text-center text-xs text-slate-400">
            After upload, processing takes about 1-2 minutes per page.
          </p>
        </div>
      </main>
    </div>
  );
}

export default UploadPage;
