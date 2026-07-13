import { useEditorStore } from "@/store/editorStore";
import { pageImageUrl } from "@/lib/api";
import { cn } from "@/lib/utils";

export function ThumbnailRail() {
  const { jobId, pagesMeta, currentPage, setCurrentPage, previewRunId } = useEditorStore();

  if (!jobId) return null;

  return (
    <aside className="no-scrollbar w-32 shrink-0 overflow-y-auto border-r border-slate-200 bg-white px-3 py-4">
      <div className="flex flex-col items-center gap-2">
        {pagesMeta.map((page) => (
          <button
            key={page.page_number}
            onClick={() => setCurrentPage(page.page_number)}
            className="flex flex-col items-center gap-1"
          >
            <div
              className={cn(
                "overflow-hidden rounded-md border-2 bg-white shadow-sm",
                page.page_number === currentPage ? "border-brand-600" : "border-slate-200",
              )}
              style={{ width: 88, height: 88 * (page.height_px / page.width_px) }}
            >
              <img
                src={pageImageUrl(jobId, page.page_number, previewRunId ? "stamped" : "source", previewRunId ?? undefined)}
                alt={`Page ${page.page_number} thumbnail`}
                className="h-full w-full object-cover"
                loading="lazy"
              />
            </div>
            <span
              className={cn(
                "text-xs",
                page.page_number === currentPage ? "font-semibold text-brand-700" : "text-slate-500",
              )}
            >
              {page.page_number}
            </span>
          </button>
        ))}
      </div>
    </aside>
  );
}
