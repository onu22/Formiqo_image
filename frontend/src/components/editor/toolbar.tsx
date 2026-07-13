import { ChevronLeft, ChevronRight, Loader2, Minus, Plus, Redo2, RefreshCw, Undo2, Upload } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useEditorStore } from "@/store/editorStore";
import { Button } from "@/components/ui/button";
import { FormiqoLogo } from "@/components/layout/logo";

export function EditorToolbar({ onExport }: { onExport: () => void }) {
  const navigate = useNavigate();
  const store = useEditorStore();
  const {
    jobId,
    pagesMeta,
    currentPage,
    zoomPct,
    fitToPage,
    dirty,
    saving,
    refreshing,
    exporting,
    history,
    future,
  } = store;

  const totalPages = pagesMeta.length;
  const docName = jobId ? `Job ${jobId.slice(0, 8)}` : "";

  function guardNavigation(action: () => void) {
    if (dirty && !window.confirm("You have unsaved changes. Leave without saving?")) return;
    action();
  }

  function confirmLeave(): boolean {
    if (!dirty) return true;
    return window.confirm("You have unsaved changes. Leave without saving?");
  }

  return (
    <header className="flex h-14 items-center gap-4 border-b border-slate-200 bg-white px-4">
      <FormiqoLogo onBeforeNavigate={confirmLeave} />
      <div className="flex items-center gap-2 border-l border-slate-200 pl-4">
        <span className="max-w-[220px] truncate text-sm font-medium text-slate-800">{docName}</span>
        {dirty && (
          <span className="flex items-center gap-1 text-xs font-medium text-amber-600">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-500" /> Unsaved changes
          </span>
        )}
      </div>

      <div className="flex items-center gap-1 border-l border-slate-200 pl-4">
        <Button variant="ghost" size="iconSm" onClick={() => store.setCurrentPage(currentPage - 1)} disabled={currentPage <= 1}>
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <span className="min-w-[92px] text-center text-sm text-slate-600">
          Page {currentPage} of {totalPages || 1}
        </span>
        <Button
          variant="ghost"
          size="iconSm"
          onClick={() => store.setCurrentPage(currentPage + 1)}
          disabled={currentPage >= totalPages}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>

      <div className="flex items-center gap-1 border-l border-slate-200 pl-4">
        <Button variant="ghost" size="iconSm" onClick={() => store.setZoomPct(zoomPct - 10)}>
          <Minus className="h-4 w-4" />
        </Button>
        <span className="w-12 text-center text-sm text-slate-600">{fitToPage ? "Fit" : `${zoomPct}%`}</span>
        <Button variant="ghost" size="iconSm" onClick={() => store.setZoomPct(zoomPct + 10)}>
          <Plus className="h-4 w-4" />
        </Button>
        <Button variant={fitToPage ? "outlineBrand" : "outline"} size="xs" onClick={() => store.setFitToPage(true)}>
          Fit
        </Button>
      </div>

      <div className="flex items-center gap-1 border-l border-slate-200 pl-4">
        <Button variant="ghost" size="iconSm" onClick={store.undo} disabled={history.length === 0} title="Undo">
          <Undo2 className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="iconSm" onClick={store.redo} disabled={future.length === 0} title="Redo">
          <Redo2 className="h-4 w-4" />
        </Button>
      </div>

      <div className="ml-auto flex items-center gap-2">
        <Button variant="outlineBrand" onClick={() => store.refreshPreview()} disabled={refreshing}>
          {refreshing ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          Refresh Preview
        </Button>
        <Button variant="outline" onClick={() => store.save()} disabled={saving || !dirty}>
          {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
          Save
        </Button>
        <Button onClick={onExport} disabled={exporting}>
          {exporting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
          Export PDF
        </Button>
      </div>

      <button
        className="ml-2 text-sm text-slate-400 hover:text-slate-600"
        onClick={() => guardNavigation(() => navigate("/"))}
        title="Back to documents"
      >
        Close
      </button>
    </header>
  );
}
