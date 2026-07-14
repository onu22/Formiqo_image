import { useEffect, useRef, useState } from "react";
import { useEditorStore } from "@/store/editorStore";
import { pageImageUrl } from "@/lib/api";
import { FieldOverlay } from "@/components/editor/field-overlay";

export function EditorCanvas() {
  const store = useEditorStore();
  const { jobId, pagesMeta, currentPage, fieldOrderByPage, fieldsById, values, selectedFieldId, zoomPct, fitToPage, previewRunId } =
    store;
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(800);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setContainerWidth(entry.contentRect.width);
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (!selectedFieldId) return;
      const active = document.activeElement;
      const tag = active?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || active?.getAttribute("contenteditable") === "true") return;

      const step = e.shiftKey ? 10 : 1;
      let dx = 0;
      let dy = 0;
      if (e.key === "ArrowUp") dy = -step;
      else if (e.key === "ArrowDown") dy = step;
      else if (e.key === "ArrowLeft") dx = -step;
      else if (e.key === "ArrowRight") dx = step;
      else return;

      e.preventDefault();
      store.nudgeField(selectedFieldId, dx, dy);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedFieldId, store]);

  const page = pagesMeta.find((p) => p.page_number === currentPage);
  if (!jobId || !page) {
    return <div className="flex flex-1 items-center justify-center text-sm text-slate-400">No page to display.</div>;
  }

  const scale = fitToPage ? Math.min((containerWidth - 48) / page.width_px, 1.6) : zoomPct / 100;
  const displayWidth = page.width_px * scale;
  const displayHeight = page.height_px * scale;
  const fieldIds = fieldOrderByPage[currentPage] ?? [];

  return (
    <div
      ref={containerRef}
      className="relative flex-1 overflow-auto bg-surface-muted px-6 py-8"
      onPointerDown={() => store.selectField(null)}
    >
      <div className="mx-auto" style={{ width: displayWidth }}>
        <div
          className="relative mx-auto bg-white shadow-md"
          style={{ width: displayWidth, height: displayHeight }}
        >
          <img
            src={pageImageUrl(jobId, currentPage, previewRunId ? "stamped" : "source", previewRunId ?? undefined)}
            alt={`Page ${currentPage}`}
            className="absolute inset-0 h-full w-full select-none"
            draggable={false}
          />
          {fieldIds.map((fieldId) => {
            const field = fieldsById[fieldId];
            if (!field) return null;
            return (
              <FieldOverlay
                key={fieldId}
                field={field}
                value={values[fieldId] ?? ""}
                scale={scale}
                selected={selectedFieldId === fieldId}
                showValueText={!previewRunId}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
