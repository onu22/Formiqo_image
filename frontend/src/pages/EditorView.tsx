import { useEffect, useState } from "react";
import { useEditorStore } from "@/store/editorStore";
import { EditorToolbar } from "@/components/editor/toolbar";
import { ThumbnailRail } from "@/components/editor/thumbnail-rail";
import { EditorCanvas } from "@/components/editor/canvas";
import { Inspector } from "@/components/editor/inspector";
import { EditorBottomBar } from "@/components/editor/bottom-bar";
import { ExportSuccessDialog } from "@/components/editor/export-success-dialog";

export function EditorView({ jobId, sourceFilename }: { jobId: string; sourceFilename: string }) {
  const store = useEditorStore();
  const { loading, loadError, fieldsById, saveError, pagesMeta } = store;
  const [exportOpen, setExportOpen] = useState(false);
  const [exportRunId, setExportRunId] = useState<string | null>(null);

  useEffect(() => {
    store.loadJob(jobId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  useEffect(() => {
    function beforeUnload(e: BeforeUnloadEvent) {
      if (store.dirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    }
    window.addEventListener("beforeunload", beforeUnload);
    return () => window.removeEventListener("beforeunload", beforeUnload);
  }, [store.dirty]);

  async function handleExport() {
    try {
      const res = await store.exportPdf();
      setExportRunId(res.download_url ? store.previewRunId : null);
      setExportOpen(true);
    } catch {
      // surfaced via store.saveError banner
    }
  }

  if (loading) {
    return <div className="flex h-screen items-center justify-center text-sm text-slate-400">Loading editor...</div>;
  }

  if (loadError) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-3 text-sm text-slate-500">
        <p>Could not load fields for this job.</p>
        <p className="text-red-600">{loadError}</p>
      </div>
    );
  }

  const fieldCount = Object.keys(fieldsById).length;

  return (
    <div className="flex h-screen flex-col">
      <EditorToolbar onExport={handleExport} />
      {saveError && (
        <div className="border-b border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">{saveError}</div>
      )}
      <div className="flex flex-1 overflow-hidden">
        <ThumbnailRail />
        <EditorCanvas />
        <Inspector />
      </div>
      <EditorBottomBar />

      <ExportSuccessDialog
        open={exportOpen}
        onClose={() => setExportOpen(false)}
        jobId={jobId}
        runId={exportRunId}
        fieldCount={fieldCount}
        pageCount={pagesMeta.length}
        sourceFilename={sourceFilename}
      />
    </div>
  );
}
